package util;

import java.util.Map;

public class StatUtil {
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
	public static void main(String[] args){
		System.out.println(Stemmer.getInstance().getStemmedForm("as"));
	}
}
