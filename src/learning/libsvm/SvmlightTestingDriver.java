package learning.libsvm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import jnisvmlight.FeatureVector;
import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;

/**
 * Driver for running trained lightsvm models against a test set
 * Reference code = http://www.mpi-inf.mpg.de/~mtb/svmlight/JNI_SVMLight_Test.java
 */
public class SvmlightTestingDriver {
	/** Binary classes for SVM */
	private static final int POSITIVE_CLASS = 1;
	private static final int NEGATIVE_CLASS = -1;
	
	/** Names of the files that will be outputted in dir */
	private static final String MAPPING_FILENAME_TEMPLATE = "%s/mapping.txt";
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";
	
	public static void main(String[] args) throws Exception {
		if(args.length != 3) {
			System.err.println("Usage: <path to train> <path to test> <models dir - must be a folder, here all the models + mapping reside>");
			return;
		}
		
		// Train is only needed to compute feature stats
		String trainFile = args[0];
		String testFile = args[1];
		String modelDir = args[2];
		if(modelDir.endsWith("/")) {
			modelDir = modelDir.substring(0, modelDir.length()-1);
		}
	    
	    VerbObjectStatComputer stats = new VerbObjectStatComputer();
	    System.out.println("Precomputing stats from train file...");
	    stats.load(trainFile);
	    System.out.println("Finished precomputation.");
	    
	    // Load all the models
	    SVMLightModel[] models = new SVMLightModel[stats.getCountDistinctVerb()];
	    for(int verbId = 1; verbId <= stats.getCountDistinctVerb(); verbId++) {
	    	models[verbId] = SVMLightModel.readSVMLightModelFromURL(
	    			new URL(String.format(MODEL_FILENAME_TEMPLATE, modelDir, verbId)));
	    }
	    
	    BufferedReader in = new BufferedReader(new FileReader(testFile));
	    
	    // filter the dataset to rows relevant to this verb, then convert to feature vectors
	    List<LabeledFeatureVector> features = new ArrayList<LabeledFeatureVector>();
	    String line = in.readLine();
	    
	    int numCorrect = 0;
	    int lineCount = 0;
	    while(line != null) {
	    	String[] toks = line.split("\t");
	    	String verb = toks[0];
			String obj = toks[1];
			// ignored - only used for stats
			int freq = Integer.parseInt(toks[2]);
			boolean isPositive = Integer.parseInt(toks[3]) == 1;
			FeatureVector featVec = SvmlightTrainingDriver.convertToLabelledFeatureVector(obj, verb, isPositive, stats);
			
			// if haven't seen the verb, just say no.
			double prediction = NEGATIVE_CLASS;
			int verbId = stats.mapVerbToId(verb);
			if(verbId >= 1) {
				prediction = models[stats.mapVerbToId(verb)].classify(featVec);
			}
			
			if(prediction > 0 == isPositive) {
				numCorrect++;
			}
			
	    	line = in.readLine();
	    	lineCount++;
	    	if(lineCount % 10000 == 0) {
	    		System.out.printf("Processed %d, acc = %d/%d = %.2f\n", lineCount, 
	    				numCorrect, lineCount, 1.0  * numCorrect / lineCount);
	    	}
	    }
	    in.close();
		System.out.printf("Processed %d, acc = %d/%d = %.2f\n", lineCount, 
				numCorrect, lineCount, 1.0  * numCorrect / lineCount);
	}
}
