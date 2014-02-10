package data.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.commons.lang3.tuple.Pair;

import util.StatUtil;

/**
 * Convert a dataset of the form "<verb> <object> <freq:int>"
 * Each word has already been porter-stemmed
 * ---- but there might be multiple rows with same VO , but their freqs aggregated
 * to rows of "<verb> <object> <freq:int> <isPositive :[0/1]>"
 * This time, the VO pairs are aggregated.
 */
public class PositiveNegativeDatasetCreator {
	public static void main(String[] args) throws Exception {
		if(args.length != 4) {
			System.err.println("Usage: <input file> <output file> <Tau: double - (MI > tau) -> pos> <K: int - num negatives for each positive>");
			return;
		}
		String inputFile = args[0];
		String outputFile = args[1];
		double TAU = Double.parseDouble(args[2]);
		int K = Integer.parseInt(args[3]);
		
		// Compute MI's
		BufferedReader in = new BufferedReader(new FileReader(inputFile));
		String line = in.readLine();
		
		long totalFreq = 0;
		int numRow = 0;
		
		Map<String, Integer> verbToCount = new HashMap<String, Integer>();
		Map<String, Integer> objToCount = new HashMap<String, Integer>();
		Map<Pair<String, String>, Integer> voToCount = new HashMap<Pair<String, String>, Integer>();
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = toks[0];
			String obj = toks[1];
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
		
		// invert objToCount
		SortedMap<Integer, List<String>> freqToObjs = new TreeMap<Integer, List<String>>();
		for(Entry<String, Integer> entry : objToCount.entrySet()) {
			String obj = entry.getKey();
			int freq = entry.getValue();
			if(!freqToObjs.containsKey(freq)) {
				freqToObjs.put(freq, new ArrayList<String>());
			}
			freqToObjs.get(freq).add(obj);
		}
		
		in.close();
		
		// Generate positive and negative
		PrintWriter out = new PrintWriter(outputFile);
		in = new BufferedReader(new FileReader(inputFile));
		line = in.readLine();
		int countPositive = 0;
		int countNegative = 0;
		
		
		for(Entry<Pair<String, String>, Integer> entry : voToCount.entrySet()) {
			String verb = entry.getKey().getLeft();
			String obj = entry.getKey().getRight();
			int freq = entry.getValue();

			double pr_vn = 1.0 * freq / totalFreq;
			double pr_v = 1.0 * verbToCount.get(verb) / totalFreq;
			double pr_n = 1.0 * objToCount.get(obj) / totalFreq;
			double mi = Math.log(pr_vn / pr_v / pr_n);
			
			if(mi > TAU) {
				// positive instance!
				out.println(verb + "\t" + obj + "\t" + freq + "\t" + 1);
				countPositive++;
				
				// create negative instances
				int numNegativeCreated = 0;
				
				int key = objToCount.get(obj);
				int listIndex = 0;
				
				while(numNegativeCreated < K) {
					// pre: guy at key + list index is a valid element & i haven't looked at it
					List<String> lst = freqToObjs.get(key);
					String possibleNPrime = lst.get(listIndex);
					
					if(!voToCount.containsKey(Pair.of(verb, possibleNPrime))) {
						// found a good negative
						numNegativeCreated++;
						countNegative++;
						out.println(verb + "\t" + possibleNPrime + "\t" + 0 + "\t" + 0);
					}
					
					if(listIndex + 1 < lst.size()) {
						listIndex++;
					} else {
						// time to search for next closest freq
						SortedMap<Integer, List<String>> view = freqToObjs.tailMap(key+1);
						if(view.size() == 0) {
							// no more nouns to look at. stop.
							break;
						} else {
							key = view.firstKey();
							listIndex = 0;
						}
					}
				}
			}
		}
		System.out.printf("Num positive : %d, Num negative : %d\n", countPositive, countNegative);
		in.close();
		out.close();
	}
}
