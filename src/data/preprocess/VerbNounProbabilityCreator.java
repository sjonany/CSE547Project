package data.preprocess;

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

/**
 * Precompute Pr(n|v) and Pr(n) from the output of RawToSubsetStemmedFilter, INCLUDING
 * v,n pairs whose MI is below threshold (this was the mistake I made previously - I only did training set stats).
 * Also makes sure that we are not computing this statistics using any test data set.
 * So, TrainTestValidationCreator must have already been run first before this class.
 * 
 * Each probability is 10^6 more than their original value. So to get the probability, divide back.
 * The output format is
 * <number of verbs>
 * verb1 [obj_i p(o_i|v_1)]* 
 * verb2 obj1 p(o1|v2) obj2 p(o2|v2) ...
 * ....
 * <number of n>
 * obj P(obj)
 */
public class VerbNounProbabilityCreator {
	/** Each probability is 10^9 more than their original value. So to get the probability, divide back. */
	public static double PROB_SCALE = 1000000000;
	public static void main(String[] args) throws Exception {
		if(args.length != 3) {
			System.err.println("Usage: <input file> <output file> <test file>");
			return;
		}
		String inputFile = args[0];
		String outputFile = args[1];
		String testFile = args[2];
		
		// Load test set into memory
		Set<Pair<String, String>> testVoPairSet = new HashSet<Pair<String,String>>();
		BufferedReader in = new BufferedReader(new FileReader(testFile));
		String line = in.readLine();
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = toks[0];
			String obj = toks[1];
			testVoPairSet.add(Pair.of(verb, obj));
			line = in.readLine();
		}
		in.close();
		
		// Go through the entire dataset and build stats, but ignoring rows corresponding to test set
		in = new BufferedReader(new FileReader(inputFile));
		line = in.readLine();
		
		long totalFreq = 0;
		int numRow = 0;
		
		Map<String, Integer> verbToCount = new HashMap<String, Integer>();
		Map<String, Integer> objToCount = new HashMap<String, Integer>();
		Map<Pair<String, String>, Integer> voToCount = new HashMap<Pair<String, String>, Integer>();
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = toks[0];
			String obj = toks[1];
			
			// Make sure not to compute stats over test set
			if(testVoPairSet.contains(Pair.of(verb, obj))) {
				line = in.readLine();
				continue;
			}
			
			int freq = Integer.parseInt(toks[2]);
			
			StatUtil.addToTally(voToCount, Pair.of(verb, obj), freq);
			StatUtil.addToTally(verbToCount, verb, freq);
			StatUtil.addToTally(objToCount, obj, freq);
			totalFreq += freq;
			
			line = in.readLine();
			numRow++;
			if(numRow % 100000 == 0) {
				System.out.printf("Processed %d rows\n", numRow);
			}
		}
		in.close();
		
		System.out.println("Clustering according to verb..");
		// I don't really want to make map of map. hopefully this saves more space and is faster.
		Map<String, List<String>> verbToObjects = new HashMap<String, List<String>>();
		for(Entry<Pair<String, String>, Integer> entry : voToCount.entrySet()) {
			String verb = entry.getKey().getLeft();
			String obj = entry.getKey().getRight();
			
			List<String> relevantObjs = verbToObjects.get(verb);
			if(relevantObjs == null) {
				relevantObjs = new ArrayList<String>();
				verbToObjects.put(verb, relevantObjs);
			}
			relevantObjs.add(obj);
		}
		
		System.out.println("Writing down stats for Pr(n|v)");
		// Write the stats down
		PrintWriter out = new PrintWriter(outputFile);
		// Pr(n|v) stats clustered by verbs
		out.println(verbToObjects.size());
		for(Entry<String, List<String>> entry : verbToObjects.entrySet()) {
			String verb = entry.getKey();
			out.print(verb + "\t");
			for(String obj : entry.getValue()) {
				int freq = voToCount.get(Pair.of(verb, obj));
				double pr_vn = 1.0 * freq / totalFreq;
				double pr_v = 1.0 * verbToCount.get(verb) / totalFreq;
				double pr_n_given_v = pr_vn / pr_v;
				out.printf("%s\t%d\t", obj, (int)(pr_n_given_v * PROB_SCALE));
			}
			out.println();
		}

		System.out.println("Writing down stats for Pr(n)");
		// Pr(n) stats
		out.println(objToCount.size());
		for(Entry<String, Integer> entry : objToCount.entrySet()) {
			String obj = entry.getKey();
			int freq = entry.getValue();
			double pr_n = 1.0 * freq / totalFreq;
			out.printf("%s\t%d\n", obj, (int)(pr_n * PROB_SCALE));
		}
		out.close();
	}
}
