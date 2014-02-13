package data.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import util.Stemmer;

/**
 * Input: Unstemmed corpus - the original, raw SVO triples
 * Output: Mapping from stemmed noun to an integer representing bit vector,
 *  where ith bit is on if noun belongs to ith cluster. The cluster mappings are from WordNet,
 *  and defined in TODO:
 */
public class StemmedNounToLexicalGroupCreator {
	public static void main(String[] args) throws Exception {
		if(args.length != 2) {
			System.err.println("Usage: <input> <output>");
			return;
		}
		String inputFile = args[0];
		String outputFile = args[1];
		
		Set<String> unstemmedNouns = new HashSet<String>();

		// Collect all unstemmed nouns in corpus.
		BufferedReader in = new BufferedReader(new FileReader(inputFile));
		String line = in.readLine();
		int totalInputRow = 0;
		while(line != null) {
			String[] toks = line.split("\t");
			String obj = Stemmer.getInstance().getStemmedForm(toks[2]);
			
			unstemmedNouns.add(obj);
			
			line = in.readLine();
			totalInputRow++;
			if(totalInputRow % 100000 == 0) {
				System.out.printf("Processed %d rows\n", totalInputRow);
			}
		}
		in.close();
		
		// Group nouns with the same stemmed form together, and union their clusters
		Map<String, Integer> stemmedNounToClusters = new HashMap<String, Integer>();
		for(String unstemmedNoun : unstemmedNouns) {
			String stemmedNoun = Stemmer.getInstance().getStemmedForm(unstemmedNoun);
			if(!stemmedNounToClusters.containsKey(stemmedNoun)) {
				stemmedNounToClusters.put(stemmedNoun, 0);
			}
			int prevMask = stemmedNounToClusters.get(stemmedNoun);
			stemmedNounToClusters.put(stemmedNoun, prevMask | getClusters(stemmedNoun));
		}
		
		// Write results to file
		PrintWriter out = new PrintWriter(outputFile);
		for(Entry<String, Integer> entry : stemmedNounToClusters.entrySet()) {
			out.println(entry.getKey() + "\t" + entry.getValue());
		}
		out.close();
	}
	
	private static int getClusters(String unstemmedNoun) {
		return 3;
	}
}
