package learning.libsvm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;

/**
 * Driver for creating svmlight models trained on given dataset.
 * Reference code = http://www.mpi-inf.mpg.de/~mtb/svmlight/JNI_SVMLight_Test.java
 */
public class SvmlightTrainingDriver {
	/** Binary classes for SVM */
	private static final int POSITIVE_CLASS = 1;
	private static final int NEGATIVE_CLASS = -1;
	
	/** Names of the files that will be outputted in dir */
	private static final String MAPPING_FILENAME_TEMPLATE = "%s/mapping.txt";
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";
	
	public static void main(String[] args) throws Exception {
		if(args.length != 2) {
			System.err.println("Usage: <path to train> <output dir - must be a folder, will output all the models + mapping>");
			return;
		}
		
		String trainFile = args[0];
		String outputDir = args[1];
		if(outputDir.endsWith("/")) {
			outputDir = outputDir.substring(0,outputDir.length()-1);
		}
	    
	    VerbObjectStatComputer stats = new VerbObjectStatComputer();
	    System.out.println("Precomputing stats from train file...");
	    stats.load(trainFile);
	    System.out.println("Finished precomputation.");
	    
	    // Write the verb to id mapping
	    PrintWriter mappingWriter = new PrintWriter(String.format(MAPPING_FILENAME_TEMPLATE, outputDir));
	    for(int verbId = 1; verbId <= stats.getCountDistinctVerb(); verbId++) { 
	    	mappingWriter.println(verbId + "\t" + stats.mapIdToVerb(verbId));
	    }
	    mappingWriter.close();
	    
	    TrainingParameters trainParam = new TrainingParameters();
	    // Switch on some debugging output
	    // http://infolab.stanford.edu/~theobald/svmlight/doc/jnisvmlight/LearnParam.html
	    trainParam.getLearningParameters().verbosity = 1;
	    trainParam.getLearningParameters().svm_costratio = 2.0;
	   
    	// For each verb, we train a single model
    	// Each such iteration requires going through the entire dataset
	    for(int verbId = 748; verbId <= stats.getCountDistinctVerb(); verbId++) {
	    	long curtime = System.currentTimeMillis();
		    BufferedReader in = new BufferedReader(new FileReader(trainFile));
		    
		    // filter the dataset to rows relevant to this verb, then convert to feature vectors
		    List<LabeledFeatureVector> features = new ArrayList<LabeledFeatureVector>();
		    String line = in.readLine();
		    while(line != null) {
		    	String[] toks = line.split("\t");
		    	String verb = toks[0];
		    	if(stats.mapVerbToId(verb) != verbId) {
		    		line = in.readLine();
		    		continue;
		    	}
				String obj = toks[1];
				// ignored - only used for stats
				int freq = Integer.parseInt(toks[2]);
				boolean isPositive = Integer.parseInt(toks[3]) == 1;
				features.add(convertToLabelledFeatureVector(obj, verb, isPositive, stats));
		    	line = in.readLine();
		    }
		    in.close();
		    
		    SVMLightModel model = new SVMLightInterface().trainModel(
		    				features.toArray(new LabeledFeatureVector[0]), trainParam);
		    model.writeModelToFile(String.format(MODEL_FILENAME_TEMPLATE, outputDir, verbId));
		    long elapsedTime = System.currentTimeMillis() - curtime;
	    	System.out.printf("%d out of %d models trained. Takes %d ms\n", verbId, stats.getCountDistinctVerb(), elapsedTime);
	    }
	}
	
	public static LabeledFeatureVector convertToLabelledFeatureVector(String obj, String verb, boolean isPositive, VerbObjectStatComputer stats) {
		double label = isPositive ? POSITIVE_CLASS : NEGATIVE_CLASS;
		// sparse representation of feature vector
		List<Integer> nonZeroIndices = new ArrayList<Integer>();
		List<Double> nonZeroVals = new ArrayList<Double>();
		for(int vid = 1; vid <= stats.getCountDistinctVerb(); vid++) {
			double val = stats.getPrNGivenV(stats.mapObjToId(obj), vid);
			if(val > 1e-5) {
				nonZeroVals.add(val);
				nonZeroIndices.add(vid);
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
}
