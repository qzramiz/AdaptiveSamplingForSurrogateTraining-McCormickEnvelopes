import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm


class IntervalBuilder:

    def __init__(self, func, config, verbose=True):
        self.config = config
        self.func = func
        self.verbose = verbose

        self.n_samples = self.config['n_samples']
        self.max_envelope_size = self.config['max_envelope_size']
        self.max_gap = self.config['max_gap']
        self.seed = self.config['seed']
        self.b = self.config['intervals']['b']
        self.a = self.config['intervals']['a']

        self.samples = None
        self.f_samples = None

        self.__init()

    def __init(self):
        self.max_interval = self.max_envelope_size * (self.b - self.a)
        self.n_samples = int(self.n_samples)
        equi_split_window = ((self.b - self.a) / self.n_samples)
        if  equi_split_window > self.max_envelope_size:
            self.n_samples = int(np.ceil((self.b - self.a )/ self.max_envelope_size))

        self.samples = np.linspace(self.a, self.b, self.n_samples)
        self.f_samples = self.func.f(self.samples)

        if self.verbose:
            print('Initial Samples : ', self.n_samples)

    def build_intervals_naive(self):
        domain_range = self.b - self.a
        self.samples = np.linspace(self.a, self.b, self.config['budget'] // 2)
        self.f_samples = self.func.f(self.samples)
        self.n_samples = self.config['budget'] // 2

        intervals = []
        for i in range(1, self.n_samples):
            a, b = self.samples[i - 1], self.samples[i]
            #         print(i,j,' - ', a,b)
            if (b - a) > self.max_interval or b < a:
                continue
            x_seg, lower, upper, gap = self.func.envelope_interval(a, b)
            intervals.append((gap.max(), a, b, x_seg, lower, upper))

        intervals.sort(key=lambda t: t[0], reverse=True)
        print('Total Intervals : ', len(intervals))
        # print([(ints[1],ints[2]) for ints in intervals])
        return intervals

    def build_intervals(self):
        intervals = []
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                a, b = self.samples[i], self.samples[j]
                #         print(i,j,' - ', a,b)
                if (b - a) > self.max_interval or b < a:
                    continue
                x_seg, lower, upper, gap = self.func.envelope_interval(a, b)
                intervals.append((gap.max(), a, b, x_seg, lower, upper))

        # Rank by gap
        intervals.sort(key=lambda t: t[0], reverse=True)
        print('Total Intervals : ', len(intervals))
        # # print('Intervals : ', intervals)
        # topN = 10
        # top_intervals = intervals  # [:topN]

        return intervals


    # Merge overlapping >60% on min length
    def merge_overlapping(self, intervals, max_interval, overlap_thresh=0.5):
        # intervals list with (gap,a,b,...)
        sorted_by_a = sorted(intervals, key=lambda t: t[1])
        merged = []
        for idx, tpl in enumerate(sorted_by_a):
            gap,a,b,*rest = tpl
            if not merged:
                merged.append([a,b,gap])
                continue
            a0,b0,g0 = merged[-1]

            overlap = max(0, min(b0,b)-max(a0,a))
            if overlap > 0:
                smaller = min(b0-a0, b-a)
                if overlap / smaller > overlap_thresh:
                    a_merged, b_merged = min(a0,a), max(b0,b)
                    if b_merged - a_merged <= max_interval:
                        merged[-1] = [min(a0,a), max(b0,b), max(g0,gap)]
                        continue
    #         print(idx, [a,b,gap])
            merged.append([a,b,gap])
        return merged

    def merge_adjust_intervals(self, df):
        intervals = df.iloc[:, 1:].values.tolist()
        intervals.sort(key=lambda x: x[0])
        n = len(intervals)

        final_intervals = intervals.copy()

        for i in range(n):
            int1 = final_intervals[i]
            for j in range(i + 1, n):
                int2 = final_intervals[j]
                if max(int1[0], int2[0]) < min(int1[1], int2[1]):
                    if int1[2] < int2[2]:
                        final_intervals[i][1] = int2[0]
                        int1[1] = int2[0]
                    else:
                        final_intervals[j][0] = int1[1]

        final_intervals = [i for i in final_intervals if i[0] < i[1]]
        intervals = []
        for a, b, _ in final_intervals:
            x_seg, lower, upper, gap = self.func.envelope_interval(a, b)
            intervals.append((gap.max(), a, b, x_seg, lower, upper))
        intervals.sort(key=lambda x: x[0], reverse=True)

        ranked = [(idx + 1, a, b, gap) for idx, (gap, a, b, _, _, _) in enumerate(intervals)]
        df = pd.DataFrame(ranked, columns=['rank', 'a', 'b', 'gap'])
        df['pc'] = (df.gap / df.gap.sum()).values

        return intervals, df


        # # Plot
        # plt.figure(figsize=(9, 9))
        # x_full = np.linspace(xL, xU, 100)
        # plt.plot(x_full, f(x_full), color="black", label="Forrester function", linewidth=2)
        #
        # cmap = cm.get_cmap('tab10', len(ranked))
        # for idx, (rank, a, b, gap) in enumerate(ranked):
        #     color = cmap(idx)
        #     x_seg, lower, upper, _ = envelope_interval(a, b)
        #     plt.plot(x_seg, lower, linestyle="--", color=color)
        #     plt.plot(x_seg, upper, linestyle="--", color=color)
        #     mid = 0.5 * (a + b)
        #     y_mid = np.interp(mid, x_seg, upper)
        #     plt.text(mid, y_mid + 0.8, f"M{rank}", color=color, ha='center', va='bottom')
        #
        # plt.title("Merged Envelopes (>60% Overlap) from LHS Intervals")
        # plt.xlabel("x")
        # plt.ylabel("f(x)")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()


    def get_intervals(self):
        # intervals = self.build_intervals()
        intervals = self.build_intervals_naive()
        # merged_ints = self.merge_overlapping(intervals, self.max_interval, self.config['merge_config']['overlap_threshold'])  # * (xU - xL))
        merged_ints = self.merge_overlapping(intervals, self.max_interval, 0.5)  # * (xU - xL))
        # Rank merged by gap desc
        merged_ints.sort(key=lambda t: t[2], reverse=True)
        ranked = [(idx + 1, a, b, gap) for idx, (a, b, gap) in enumerate(merged_ints)]
        merged_ints.sort(key=lambda t: t[0], reverse=False)

        df = pd.DataFrame({
            "Merged Rank": [r for r, _, _, _ in ranked],
            "a": [round(a, 3) for _, a, _, _ in ranked],
            "b": [round(b, 3) for _, _, b, _ in ranked],
            "max_gap": [round(gap, 3) for *_, gap in ranked]
        })

        merged_intervals, merged_df = self.merge_adjust_intervals(df)

        return merged_intervals, merged_df


    def plot(self, ranked):
        # Plot
        plt.figure(figsize=(9, 5))
        x_full = np.linspace(self.a, self.b, 2000)
        plt.plot(x_full, self.func.f(x_full), color="black", label="Forrester function", linewidth=2)

        cmap = cm.get_cmap('tab10', len(ranked))
        for idx, (rank, a, b, gap) in enumerate(ranked):
            color = cmap(idx)
            x_seg, lower, upper, _ = self.func.envelope_interval(a, b)
            plt.plot(x_seg, lower, linestyle="--", color=color)
            plt.plot(x_seg, upper, linestyle="--", color=color)
            mid = 0.5 * (a + b)
            y_mid = np.interp(mid, x_seg, upper)
            plt.text(mid, y_mid + 0.8, f"M{rank}", color=color, ha='center', va='bottom')

        plt.title("Merged Envelopes (>60% Overlap) from LHS Intervals")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # # Show table
        # import pandas as pd
        # df = pd.DataFrame({
        #     "Merged Rank": [r for r, _, _, _ in ranked],
        #     "a": [round(a, 3) for _, a, _, _ in ranked],
        #     "b": [round(b, 3) for _, _, b, _ in ranked],
        #     "max_gap": [round(gap, 3) for *_, gap in ranked]
        # })
        # tools.display_dataframe_to_user("Merged intervals (>60% overlap)", df)
