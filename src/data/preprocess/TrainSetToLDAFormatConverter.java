package data.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.lang3.tuple.Pair;

import util.StatUtil;

/**
 * Convert a training set created by TrainTestValidationCreator
 * into a format that will be accepted by MCMC.R code
 * Generates 3 files:
 * 1. verbIdx.txt - rows of stemmed english verbs sorted alphabetically 
 * 2. nounIdx.txt - rows of stemmed english nouns sorted alphabetically
 * 3. vnIdx.txt - rows of verbId:int , nounId:int sorted w.r.t verbId, then nounId
 * 4. vnIdxSmall.txt - a subset of vnIdx.txt - just (v,n) pairs where v is one of the top 10 most frequent verbs.
 * The ids start at 1.
 * 
 * @author sjonany
 */
public class TrainSetToLDAFormatConverter {
	// number of verbs in the subset
	private static final int TOP_K = 10;
	public static void main(String[] args) throws Exception {
		String inputPath = args[0];
		String outputDir = args[1];
		
		TreeSet<String> nouns = new TreeSet<String>();
		TreeSet<String> verbs = new TreeSet<String>();
		Map<Pair<String, String>, Integer> vnToCount = new HashMap<Pair<String, String>, Integer>();
		final Map<String, Integer> verbToClusterSize = new HashMap<String, Integer>();
		
		BufferedReader in = new BufferedReader(new FileReader(inputPath));
		String line = in.readLine();
		int lineCount = 0;
		while(line != null) {
			lineCount++;
			if(lineCount % 10000 == 0) {
				System.out.println(lineCount);
			}
			String[] toks = line.split("\t");
			String verb = toks[0];
			String obj = toks[1];
			int freq = Integer.parseInt(toks[2]);
			boolean isPositive = Integer.parseInt(toks[3]) == 1;
			
			// We want to recover the generative model for only the positive samples
			if(!isPositive) {
				line = in.readLine();
				continue;
			}
			
			// v,o pairs are distinct, so no need to accumulate
			vnToCount.put(Pair.of(verb, obj), freq);
			nouns.add(obj);
			verbs.add(verb);
			StatUtil.addToTally(verbToClusterSize, verb, 1);
			line = in.readLine();
		}
		
		PrintWriter verbWriter = new PrintWriter(outputDir + "verbIdx.txt");
		PrintWriter nounWriter = new PrintWriter(outputDir + "nounIdx.txt");
		PrintWriter vnWriter = new PrintWriter(outputDir + "vnIdx.txt");
		PrintWriter vnSubsetWriter = new PrintWriter(outputDir + "vnIdxSmall.txt");
		
		final Map<String, Integer> nounToId = new HashMap<String, Integer>();
		final Map<String, Integer> verbToId = new HashMap<String, Integer>();
		
		for(String noun : nouns) {
			nounToId.put(noun, nounToId.size() + 1);
			nounWriter.println(noun);
		}
		
		for(String verb : verbs) {
			verbToId.put(verb, verbToId.size() + 1);
			verbWriter.println(verb);
		}
		
		List<Pair<String, String>> vnPairs = new ArrayList<Pair<String, String>>();
		for(Pair<String, String> pair : vnToCount.keySet()) {
			vnPairs.add(pair);
		}
		
		// sort by verb id, then noun id
		Collections.sort(vnPairs, new Comparator<Pair<String, String>>() {
			@Override
      public int compare(Pair<String, String> arg0, Pair<String, String> arg1) {
	      int v1 = verbToId.get(arg0.getLeft());
	      int v2 = verbToId.get(arg1.getLeft());
	      
	      if(v1 != v2) {
	      	return v1 - v2;
	      }

	      int n1 = nounToId.get(arg0.getRight());
	      int n2 = nounToId.get(arg1.getRight());
	      
	      return n1 - n2;
      }
		});
		
		List<Entry<String, Integer>> entries = new ArrayList<Entry<String, Integer>>();
		for(Entry<String, Integer> entry : verbToClusterSize.entrySet()) {
			entries.add(entry);
		}
		
		// Sort in descending order of # training instances
		Collections.sort(entries, new Comparator<Entry<String, Integer>>() {
			@Override
      public int compare(Entry<String, Integer> arg0,
          Entry<String, Integer> arg1) {
	      return -arg0.getValue() + arg1.getValue();
			}
		});
		
		Set<String> verbSubset = new HashSet<String>();
		for(int i = 0; i < TOP_K; i++) {
			verbSubset.add(entries.get(i).getKey());
		}
		System.out.println("Verb subset = " + verbSubset);
		
		for(Pair<String, String> vnPair : vnPairs) {
			String str = verbToId.get(vnPair.getKey()) + "," + nounToId.get(vnPair.getValue());
			for(int count = 0; count < vnToCount.get(vnPair); count++) {
				vnWriter.println(str);
			}
		}
		
		for(Pair<String, String> vnPair : vnPairs) {
			String str = verbToId.get(vnPair.getKey()) + "," + nounToId.get(vnPair.getValue());
			for(int count = 0; count < vnToCount.get(vnPair); count++) {
				vnWriter.println(str);
			}
			if(verbSubset.contains(vnPair.getKey())) {
				for(int count = 0; count < vnToCount.get(vnPair); count++) {
					vnSubsetWriter.println(str);
				}
			}
		}
		
		in.close();
		verbWriter.close();
		nounWriter.close();
		vnWriter.close();
		vnSubsetWriter.close();
	}
}
