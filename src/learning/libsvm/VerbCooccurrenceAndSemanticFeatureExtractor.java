package learning.libsvm;

import java.util.ArrayList;
import java.util.List;

import jnisvmlight.LabeledFeatureVector;

/**
 * Feature extractor that only uses verb coocurence as feature
 */
public class VerbCooccurrenceAndSemanticFeatureExtractor implements FeatureExtractor {
	private static final int POSITIVE_CLASS = 1;
	private static final int NEGATIVE_CLASS = -1;
	// Each bit vector will be of this length, a 1 indicates that the noun 
	private static final int NUMBER_OF_NOUN_CLUSTERS = 26;
	
	/** Important stats needed to compute the feature vector. */
	private VerbObjectStatComputer stats;
	
	public VerbCooccurrenceAndSemanticFeatureExtractor(VerbObjectStatComputer stats) {
		this.stats = stats;
	}
	
	@Override
  public LabeledFeatureVector convertDataPointToFeatureVector(String verb,
      String object, boolean isPositive) {
		double label = isPositive ? POSITIVE_CLASS : NEGATIVE_CLASS;
		// sparse representation of feature vector
		List<Integer> nonZeroIndices = new ArrayList<Integer>();
		List<Double> nonZeroVals = new ArrayList<Double>();
		for(int vid = 1; vid <= stats.getCountDistinctVerb(); vid++) {
			double val = stats.getPrNGivenV(stats.mapObjToId(object), vid);
			if(val > 1e-5) {
				nonZeroVals.add(val);
				nonZeroIndices.add(vid);
			}
		}
		
		// Convert bit vector to feature
		// Need to offset the feature index, since the feat indices before this, we have verbs
		int featureOffset = stats.getCountDistinctVerb();
		int clusterBits = getCluster(object);
		for(int i = 0; i < NUMBER_OF_NOUN_CLUSTERS; i++) {
			if((clusterBits & (1<<i)) != 0) {
				nonZeroIndices.add(featureOffset + i + 1);
				nonZeroVals.add(1.0);
			}
		}
		
		// Ugh. library will complain if all 0
		if(nonZeroVals.size() == 0) {
			// http://searchcode.com/codesearch/view/61770141
			nonZeroVals.add(0.0);
			nonZeroIndices.add(1);
		}
		
		int[] dims = new int[nonZeroIndices.size()];
		double[] vals = new double[dims.length];
		for(int i = 0; i < dims.length; i++) {
			dims[i] = nonZeroIndices.get(i);
			vals[i] = nonZeroVals.get(i);
		}
		LabeledFeatureVector feature = new LabeledFeatureVector(label, dims, vals);
		return feature;
  }
	
	private int getCluster(String noun) {
		// TODO: make another stats obj
		return 3;
	}
}
