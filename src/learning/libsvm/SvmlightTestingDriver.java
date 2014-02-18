package learning.libsvm;

import java.io.File;
import java.net.URL;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
import util.StatUtil.ClassificationPerformance;
import util.SvmlightUtil;
import util.WordnetCluster;

/**
 * Driver for running trained lightsvm models against a test set Reference code
 * = http://www.mpi-inf.mpg.de/~mtb/svmlight/JNI_SVMLight_Test.java
 */
public class SvmlightTestingDriver {
	/** Names of the files that will be outputted in dir */
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";

	public static void main(String[] args) throws Exception {
		if (args.length != 4) {
			System.err
			    .println("Usage: <path to stat> <path to test> <models dir - must be a folder, here all the models + mapping reside> <wordnet path>");
			return;
		}

		String statFile = args[0];
		String testFile = args[1];
		String modelDir = args[2];
		String wordnetPath = args[3];
		if (modelDir.endsWith("/")) {
			modelDir = modelDir.substring(0, modelDir.length() - 1);
		}

		VerbObjectStatComputer stats = new VerbObjectStatComputer();
		System.out.println("Precomputing stats from train file...");
		stats.load(statFile);
		System.out.println("Finished precomputation.");

		WordnetCluster wordnet = new WordnetCluster(wordnetPath);
	  FeatureExtractor featureExtractor = new VerbCooccurrenceAndSemanticFeatureExtractor(stats, wordnet);
		
	  ClassificationPerformance globalResult = new ClassificationPerformance();
	  // Perform test per verb - Assumes that we are only testing on verbs that we have trained on
	  for(int verbId = 1; verbId <= stats.getCountDistinctVerb(); verbId++) {
	  	String verbString = stats.mapIdToVerb(verbId);
	  	// Load all the test cases for this verb
	  	LabeledFeatureVector[] testSet = SvmlightUtil.filterDatasetToVerb(testFile, stats.mapIdToVerb(verbId), featureExtractor);
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
	  	ClassificationPerformance localResult = SvmlightUtil.testModel(model, testSet);
	  	
	  	globalResult.merge(localResult);

			System.out.printf("Processed %d/%d verbs\nLocal result: acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n", 
					verbId, stats.getCountDistinctVerb(),
			    localResult.tp + localResult.tn, localResult.getDatasetSize(), localResult.getAccuracy(), localResult.getPrecision(),
			    localResult.getRecall(), localResult.getFscore());
	  	
			System.out.printf("Global result: acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n",
			    globalResult.tp + globalResult.tn, globalResult.getDatasetSize(), globalResult.getAccuracy(), globalResult.getPrecision(),
			    globalResult.getRecall(), globalResult.getFscore());
	  }

		System.out.printf("Processed all! , acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n",
		    globalResult.tp + globalResult.tn, globalResult.getDatasetSize(), globalResult.getAccuracy(), globalResult.getPrecision(),
		    globalResult.getRecall(), globalResult.getFscore());
	}
}
