package learning.libsvm;

import jnisvmlight.LabeledFeatureVector;

/**
 * Interface for a feature extractor.
 */
public interface FeatureExtractor {
	/** Binary classes for SVM */
	public static final int POSITIVE_CLASS = 1;
	public static final int NEGATIVE_CLASS = -1;
	
	/**
	 * Input : data point
	 * Output : labelled feature vector based on the data point
	 */
	public LabeledFeatureVector convertDataPointToFeatureVector(String verb, String object, boolean isPositive);
}
