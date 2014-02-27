package util;

import java.util.HashMap;
import java.util.Map;

public class StatUtil {
	public static void main(String[] args) {
		Map<Integer, Integer> tally = new HashMap<Integer, Integer> ();
		double[] weights = {1, 2, 3};
		for(int i = 0; i < 6000; i++) {
			addToTally(tally, sampleMultinomial(weights), 1);
		}
		System.out.println(tally);
	}
	
	/**
	 * O(N) multinomial sampler
	 * @return zero-indexed i, with probability weights[i] / sum{weights}
	 */
	public static int sampleMultinomial(double[] weights) {
		double[] sum = new double[weights.length];
		sum[0] = weights[0];
		for(int i = 1; i < weights.length; i++) {
			sum[i] = weights[i] + sum[i-1];
		}
		
		double rand = Math.random() * sum[sum.length-1];
    for (int i = 0; i < weights.length; i++) {
        if (rand <= sum[i])
           return i;
    }
    return weights.length-1;
	}
	
	public static <K> void addToTally(Map<K, Integer> tally, K key, int val) {
		if(!tally.containsKey(key)) {
			tally.put(key, val);
		} else {
			tally.put(key, tally.get(key) + val);
		}
	}
	
	public static class ClassificationPerformance {
		// true/false pos/negs
		public int tp;
		public int fp;
		public int fn;
		public int tn;
		
		public ClassificationPerformance() {
			tp = fp = fn = tn = 0;
		}
		
		public double getAccuracy() {
			return 1.0 * (tp + tn) / getDatasetSize();
		}
		
		public double getPrecision() {
			return 1.0 * tp / (tp + fp);
		}
		
		public double getRecall() {
			return 1.0 * tp / (tp + fn);
		}
		
		public double getFscore() {
			double p = getPrecision();
			double r = getRecall();
			return 2.0 * p * r / (p + r);
		}
		
		public int getDatasetSize() {
			return tp + fp + fn + tn;
		}
		
		/**
		 * Merge stats with another classification performance.
		 */
		public void merge(ClassificationPerformance other) {
			this.tp += other.tp;
			this.fp += other.fp;
			this.fn += other.fn;
			this.tn += other.tn;
		}
	}
}
