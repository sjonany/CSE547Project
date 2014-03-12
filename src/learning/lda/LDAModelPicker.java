package learning.lda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.lang3.tuple.Pair;

import util.StatUtil;
import util.StatUtil.ClassificationPerformance;

/**
 * Given a collection of LDA models obtained from Gibbs sampling, choose
 * one that has the highest performance on the validation set.
 */
public class LDAModelPicker {

	private static final Pattern THETA_MODEL_PATTERN = Pattern.compile("thetaAtIter(\\d*).txt");
	private static final Pattern BETA_MODEL_PATTERN = Pattern.compile("betaAtIter(\\d*).txt");
	
	public static void main(String[] args) throws Exception {
		if(args.length != 3) {
			System.err.println("Usage: <modelDir> <testPath> <genCorpusPath>");
			return;
		}
		
		String modelDir = args[0];
		if(!modelDir.endsWith("/")) modelDir += "/";
		String testPath = args[1];
		/** genCorpus will be used to compute P(D) and P(a1| D) */
		String genCorpusPath = args[2];

		// Precompute the prior infos needed to convert LDA judgment to discriminative
		// count(noun ^ distractor) 
		Map<String, Integer> countNAndD = new HashMap<String, Integer>();
		long countDistractor = 0;
		long countData = 0;
		
		BufferedReader genReader = new BufferedReader(new FileReader(genCorpusPath));
		String line = genReader.readLine();
		int lineCount = 0;
		while(line != null) {
    	String[] toks = line.split("\t");
    	String obj = toks[1];
    	boolean isPositive = Integer.parseInt(toks[3]) == 1;
    	int freq = 1;
    	if(!isPositive) {
    		countDistractor += freq;
    		StatUtil.addToTally(countNAndD, obj, freq);
    	}
    	countData += freq;
			line = genReader.readLine();
			lineCount++;
			if(lineCount % 100000 == 0) {
				System.out.printf("Processed %d lines from generalization corpus. \n", lineCount);
			}
		}
		Map<String, Double> prNGivenD = new HashMap<String, Double>();
		for(Entry<String, Integer> entry : countNAndD.entrySet()) {
			prNGivenD.put(entry.getKey(), 1.0 * entry.getValue() / countDistractor);
		}
		double prD = 1.0 * countDistractor / countData;
		genReader.close();
		System.out.println("Finished precomputing priors from generalization corpus.");
		
		Map<Integer, String> thetaFiles = new HashMap<Integer, String>();
		Map<Integer, String> betaFiles = new HashMap<Integer, String>();
		
		// First, see which models are present
		for(File file : new File(modelDir).listFiles()) {
			String name = file.getName();
			
			{
				Matcher matcher = THETA_MODEL_PATTERN.matcher(name);
				boolean matches = matcher.matches();
				if(matches) {
					int iterNum = Integer.parseInt(matcher.group(1));
					thetaFiles.put(iterNum, name);
				}
			}
			{
				Matcher matcher = BETA_MODEL_PATTERN.matcher(name);
				boolean matches = matcher.matches();
				if(matches) {
					int iterNum = Integer.parseInt(matcher.group(1));
					betaFiles.put(iterNum, name);
				}
			}
		}
		System.out.println(thetaFiles);
		System.out.println(betaFiles);
		double maxF1 = Double.NEGATIVE_INFINITY;
		int bestModelId = -1;
		for(int modelId : thetaFiles.keySet()) {
			LDAModel model = LDAModel.loadModel(modelDir, modelDir + thetaFiles.get(modelId), modelDir + betaFiles.get(modelId));
			double f1 = calcF1(model, prNGivenD, prD, testPath);
			System.out.printf("modelId %d, f1 = %.6f \n", modelId, f1);
			if(f1 > maxF1) {
				maxF1 = f1;
				bestModelId = modelId;
			}
		}
		
		System.out.printf("Best model id = %d, f1 = %.6f\n", bestModelId, maxF1);
	}
	
	/**
	 * Calculate best f1 score for this model, by varying thresholds against the validation set.
	 */
	private static final double calcF1(LDAModel model, Map<String, Double> prNGivenD, double prD, String testPath) throws IOException {
		// List of all the test points, where the boolean = true if gold is positive
	  // and the double is the w.x value returned by the lda model for that point
	  List<Pair<Double, Boolean>> gradedTestPoint = new ArrayList<Pair<Double, Boolean>>();
	  Set<Double> thresholdVals = new HashSet<Double>();
		
		BufferedReader testReader = new BufferedReader(new FileReader(testPath));
		int lineCount = 0;
		String line = testReader.readLine();
    while(line != null) {
    	lineCount++;
    	if(lineCount % 100000 == 0)
    		System.out.println(lineCount);
    	String[] toks = line.split("\t");
    	String verb = toks[0];
    	
    	String obj = toks[1];
    	boolean isPositiveGold = Integer.parseInt(toks[3]) == 1;
    	double isValidPr = predictIsNotDistractor(model, verb, obj, prNGivenD, prD);
    	gradedTestPoint.add(Pair.of(isValidPr, isPositiveGold));
    	thresholdVals.add(isValidPr);
    	line = testReader.readLine();
    }
		testReader.close();
		
		double maxFScore = -1;
	 	// Just doing a simple N^2 way instead of sort then linear scan since samples are small enough.
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
			if(!Double.isNaN(result.getFscore())) {
				maxFScore = Math.max(maxFScore, result.getFscore());
			}
		}
		return maxFScore;
	}
	
	private static double predictIsNotDistractor(LDAModel model, String verb, String obj, Map<String, Double> prNGivenD,
			double prD) {
		double pLda = 0.0;
		for(int topic = 0; topic < model.getTopicCount(); topic++) {
			pLda += model.getPrNounForTopic(obj, topic) * model.getPrTopicForVerb(topic, verb);
		}
		double prN_D = prNGivenD.containsKey(obj) ? prNGivenD.get(obj) : 0.0;
		double num = (1.0 - prD)  * pLda;
		double den = prD * prN_D + num;
		return num / den;
	}
}
