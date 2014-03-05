package learning.libsvm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;

import org.apache.commons.lang3.tuple.Pair;

import util.StatUtil.ClassificationPerformance;
import util.SvmlightUtil;
import util.WordnetCluster;

/**
 * Vary the intercepts for an SVM model to produce points 
 * for a precision / recall graph. 
 * Also useful to pick the best intercept to use against the test set.
 * 
 * Where by threshold, 
 * if (w * x > threshold) -> choose positive class
 * 
 * Output: a text file where each row is
 * <threshold value: double> \t <tp:int> \t <fp:int> \t <fn:int> \t <tn:int> 
 * \t <precision:double> \t <recall:double> \t <fscore:double>
 */
public class VaryingInterceptDriver {
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";
	
	private static final String SEMANTIC = "semantic";
	private static final String VERB = "verb";
	private static final String VERB_AND_SEMANTIC = "verbAndSemantic";
	
	public static void main(String[] args) throws Exception {
		if (args.length != 7) {
			System.err
			    .println("Usage: <path to stat> <path to test>" +
			    		" <models dir - must be a folder, here all the models + mapping reside> "
			    		 + " <wordnet path> <path to verb to test> <path to output text file> <model type: semantic/verb/verbAndSemantic>");
			return;
		}

		String statFile = args[0];
		String testFile = args[1];
		String modelDir = args[2];
		String wordnetPath = args[3];
		String verbToTestPath = args[4];
		String outputFile = args[5];
		String modelType = args[6];
		
		if (modelDir.endsWith("/")) {
			modelDir = modelDir.substring(0, modelDir.length() - 1);
		}

		VerbObjectStatComputer stats = new VerbObjectStatComputer();
		System.out.println("Precomputing stats from train file...");
		stats.load(statFile);
		System.out.println("Finished precomputation.");

		WordnetCluster wordnet = new WordnetCluster(wordnetPath);
	  FeatureExtractor featureExtractor = null;
	  if(modelType.equals(SEMANTIC)) {
	  	featureExtractor = new SemanticFeatureExtractor(wordnet);
	  } else if(modelType.equals(VERB_AND_SEMANTIC)) {
	  	featureExtractor = new VerbCooccurrenceAndSemanticFeatureExtractor(stats, wordnet);
	  } else if(modelType.equals(VERB)) {
	  	featureExtractor = new VerbCooccurrenceFeatureExtractor(stats);
	  } else {
	  	System.err.println("Unrecognized model type = " + modelType);
	  	return;
	  }
		
	  ClassificationPerformance globalResult = new ClassificationPerformance();

	  // List of all the test points, where the boolean = true if gold is positive
	  // and the double is the w.x value returned by the svm model for that point
	  List<Pair<Double, Boolean>> gradedTestPoint = new ArrayList<Pair<Double, Boolean>>();
		TreeSet<Double> thresholdVals = new TreeSet<Double>();
	  
		BufferedReader verbToTestReader = new BufferedReader(new FileReader(verbToTestPath));
		String line = verbToTestReader.readLine();
		int lineCount = 0;
		while(line != null) {
			// Perform test for each verb
	  	String verbString = line.split("\t")[0];
	  	int verbId = stats.mapVerbToId(verbString);
	  	// Load all the test cases for this verb
	  	LabeledFeatureVector[] testSet = SvmlightUtil.filterDatasetToVerb(testFile, verbString, featureExtractor);
	  	if(testSet.length == 0) {
	  		continue;
	  	}
	  	
			String file = String.format(MODEL_FILENAME_TEMPLATE, modelDir, verbId);
			if(!new File(file).exists()) {
				System.err.println("No model for verbId = " + verbId + " = " + verbString + 
						", even though there are test cases for it");
				continue;
			}
			SVMLightModel model = SVMLightModel.readSVMLightModelFromURL(new URL("file:" + file));
			for(LabeledFeatureVector testPoint : testSet) {
				double prediction = model.classify(testPoint);
				boolean isPositive = testPoint.getLabel() > 0;
				gradedTestPoint.add(Pair.of(prediction, isPositive));
				thresholdVals.add(prediction);
			}
			
			lineCount++;
			line = verbToTestReader.readLine();
	  }
		
		verbToTestReader.close();
		System.out.println("Processed all test points. Now varying the threshold values.");
		
		// Just doing a simple N^2 way instead of sort then linear scan since samples are small enough.
		PrintWriter resultWriter = new PrintWriter(outputFile);
		resultWriter.println("Threshold\tTP\tFP\tFN\tTN\tprecision\trecall\tfscore");
		for(double threshold : thresholdVals) {
			ClassificationPerformance result = new ClassificationPerformance();
			for(Pair<Double, Boolean> point : gradedTestPoint) {
				boolean predictIsPositive = point.getKey() > threshold;
				boolean isPositive = point.getValue();
				
				if(isPositive) {
					if(predictIsPositive) {
						result.tp++;
					} else {
						result.fn++;
					}
				} else {
					if(predictIsPositive) {
						result.fp++;
					} else {
						result.tn++;
					}
				}
			}
			resultWriter.printf("%.6f\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\n", 
					threshold, result.tp, result.fp, result.fn, result.tn, result.getPrecision(), result.getRecall(), result.getFscore());
		}
		resultWriter.close();
	}
}
