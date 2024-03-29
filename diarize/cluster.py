from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

class ClusterModule():

    def __init__(self, cfg):
        self.cfg = cfg

    def cluster(self, embeddings, starts, ends):
        # Apply AgglomerativeClustering

        if self.cfg.cluster.normalize:
            embeddings = normalize(embeddings, axis=1, norm='l2')
        num_cluster = self.cfg.cluster.num_cluster
        if num_cluster == 'None':
            cluster_labels = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=self.cfg.cluster.threshold).fit_predict(
                    embeddings)
        else:
            cluster_labels = AgglomerativeClustering(
                n_clusters=int(num_cluster),
                linkage='average',
                metric='cosine').fit_predict(embeddings)

        SEC_tuples = [(s, e, c)
                      for s, e, c in zip(starts, ends, cluster_labels)]

        if not self.cfg.vad.merge_vad:
            SEC_tuples = self.merge_speakers(SEC_tuples)

        return SEC_tuples

    def merge_speakers(self, SEC_tuples):
        # Merge adjacent segments with same speakers
        win_length = self.cfg.embedding.win_length
        hop_length = self.cfg.embedding.hop_length
        prev_label, prev_start, prev_end = -1, -1, -1
        overlap = round(float(win_length) - float(hop_length), 3)
        output = []

        for start, end, label in SEC_tuples:
            if prev_label >= 0:
                if prev_label != label or start > prev_end:
                    if start >= prev_end:
                        output.append(
                            (prev_start, max(prev_end, prev_start), prev_label))
                        prev_start = start
                        prev_end = end
                    else:
                        output.append(
                            (prev_start, max(prev_end - overlap / 2, prev_start),
                             prev_label
                             ))  # Split into half and assign different speaker
                        prev_start = start + overlap / 2
                        prev_end = end
                else:
                    prev_end = end
            else:
                prev_start = start
                prev_end = end

            prev_label = label

        # append the last tuple
        if prev_end > prev_start:
            output.append((prev_start, prev_end, prev_label))

        # round the numbers up to 3 decmial
        for i in range(len(output)):
            output[i] = (round(output[i][0], 3), round(output[i][1],
                                                       3), output[i][2])
        return output
