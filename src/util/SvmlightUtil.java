package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import util.StatUtil.ClassificationPerformance;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
import learning.libsvm.FeatureExtractor;

/**
 * Utility functions for learning procedures related to Svmlight
 */
public class SvmlightUtil {
	/**
	 * Run the trained model against a test set and report the performance
	 */
	public static ClassificationPerformance testModel(SVMLightModel model, LabeledFeatureVector[] testSet, double threshold) {
		ClassificationPerformance result = new ClassificationPerformance();
		for(LabeledFeatureVector testPoint : testSet) {
			double prediction = model.classify(testPoint);
			boolean isPositive = testPoint.getLabel() > 0;
			
			if(isPositive) {
				if(prediction > threshold) {
					result.tp++;
				} else {
					result.fn++;
				}
			} else {
				if(prediction > threshold) {
					result.fp++;
				} else {
					result.tn++;
				}
			}
		}
		return result;
	}
	
	public static ClassificationPerformance testModel(SVMLightModel model, LabeledFeatureVector[] testSet) {
		return testModel(model, testSet, 0.0);
	}
	
	/**
	 * Read the entire dataset from 'file', but only consider rows which corresponds to 'targetVerb'.
	 * For each relevant row, convert it into a labeled feature vector.
	 * The returned value is the compilation of all such vectors.
	 */
	public static LabeledFeatureVector[] filterDatasetToVerb(String file, String targetVerb, FeatureExtractor featureExtractor) throws Exception {
		BufferedReader in = new BufferedReader(new FileReader(file));
		int countPos = 0;
		int countNeg = 0;
		
    // filter the dataset to rows relevant to this verb, then convert to feature vectors
    List<LabeledFeatureVector> features = new ArrayList<LabeledFeatureVector>();
    String line = in.readLine();
    int lineCount = 0;
    while(line != null) {
    	lineCount++;
    	if(lineCount % 100000 == 0)
    		System.out.println(lineCount);
    	String[] toks = line.split("\t");
    	String verb = toks[0];
    	if(!verb.equals(targetVerb)) {
    		line = in.readLine();
    		continue;
    	}
		String obj = toks[1];
		boolean isPositive = Integer.parseInt(toks[3]) == 1;
		if(isPositive) {
			countPos++;
		} else {
			countNeg++;
		}
		features.add(featureExtractor.convertDataPointToFeatureVector(verb, obj, isPositive));
    	line = in.readLine();
    }
    in.close();
	  System.out.printf("Positive = %d, Negative = %d\n", countPos, countNeg);
    return features.toArray(new LabeledFeatureVector[0]); 
	}
}
