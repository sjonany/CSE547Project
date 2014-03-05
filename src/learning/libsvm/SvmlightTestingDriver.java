package learning.libsvm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
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

	private static final String SEMANTIC = "semantic";
	private static final String VERB = "verb";
	private static final String VERB_AND_SEMANTIC = "verbAndSemantic";
	public static void main(String[] args) throws Exception {
		if (args.length != 7) {
			System.err
			    .println("Usage: <path to stat> <path to test>" +
			    		" <models dir - must be a folder, here all the models + mapping reside> "
			    		 + " <wordnet path> <path to verb to test> <model type: semantic/verb/verbAndSemantic> <threshold: double>");
			return;
		}

		String statFile = args[0];
		String testFile = args[1];
		String modelDir = args[2];
		String wordnetPath = args[3];
		String verbToTestPath = args[4];
		String modelType = args[5];
		double threshold = Double.parseDouble(args[6]);
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
	  	ClassificationPerformance localResult = SvmlightUtil.testModel(model, testSet, threshold);
	  	
	  	globalResult.merge(localResult);

			System.out.printf("Processed %d verbs\nLocal result: acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n", 
					lineCount+1,
			    localResult.tp + localResult.tn, localResult.getDatasetSize(), localResult.getAccuracy(), localResult.getPrecision(),
			    localResult.getRecall(), localResult.getFscore());
	  	
			System.out.printf("Global result: acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n",
			    globalResult.tp + globalResult.tn, globalResult.getDatasetSize(), globalResult.getAccuracy(), globalResult.getPrecision(),
			    globalResult.getRecall(), globalResult.getFscore());
			
			lineCount++;
			line = verbToTestReader.readLine();
	  }
		
		verbToTestReader.close();
		System.out.printf("Processed all! , acc = %d/%d = %.6f, precision = %.6f, recall = %.6f, f1 = %.6f\n",
		    globalResult.tp + globalResult.tn, globalResult.getDatasetSize(), globalResult.getAccuracy(), globalResult.getPrecision(),
		    globalResult.getRecall(), globalResult.getFscore());
	}
}
