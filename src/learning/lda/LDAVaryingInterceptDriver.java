package learning.lda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;

import util.StatUtil;
import util.StatUtil.ClassificationPerformance;

/**
 * Vary the decision threshold value for the discriminative version of LDA
 * and give precision recall points 
 */
public class LDAVaryingInterceptDriver {
	public static void main(String[] args) throws Exception {
		String modelDir = args[0];
		String testPath = args[1];
		/** genCorpus will be used to compute P(D) and P(a1| D) */
		String genCorpusPath = args[2];
		String outputFile = args[3];

		LDAModel model = LDAModel.loadModel(modelDir);
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
    	// TODO: How to compute P(a1|D) ? since frequency is undefined? I just treat all freqs as 1 then.
    	// int freq = Integer.parseInt(toks[2]);
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
		
		BufferedReader testReader = new BufferedReader(new FileReader(testPath));
		lineCount = 0;
		line = testReader.readLine();

	  // List of all the test points, where the boolean = true if gold is positive
	  // and the double is the w.x value returned by the lda model for that point
	  List<Pair<Double, Boolean>> gradedTestPoint = new ArrayList<Pair<Double, Boolean>>();
	  Set<Double> thresholdVals = new HashSet<Double>();
    while(line != null) {
    	lineCount++;
    	if(lineCount % 100000 == 0)
    		System.out.println(lineCount);
    	String[] toks = line.split("\t");
    	String verb = toks[0];
    	
    	String obj = toks[1];
    	boolean isPositiveGold = Integer.parseInt(toks[3]) == 1;
    	double isValidPr = getPrIsNotDistractor(model, verb, obj, prNGivenD, prD);
    	gradedTestPoint.add(Pair.of(isValidPr, isPositiveGold));
    	thresholdVals.add(isValidPr);
    	
    	line = testReader.readLine();
    }
    testReader.close();
    
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
	
	private static double getPrIsNotDistractor(LDAModel model, String verb, String obj, Map<String, Double> prNGivenD,
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
