package data.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import util.StatUtil;
import util.Stemmer;

/**
 * Input: SVO triple dataset http://rtw.ml.cmu.edu/resources/svo/
 * Output: A subset of VO rows of popular verbs and nouns, and they have
 * 	been porter-stemmed http://tartarus.org/martin/PorterStemmer/
 */
public class RawToSubsetStemmedFilter {
	private static final Stemmer stemmer = new Stemmer();
	
	public static void main(String[] args) throws Exception {
		if(args.length != 4) {
			System.err.println("Usage: <input file> <output file> <verb threshold: int> <obj threshold: int>");
			return;
		}
		String inputFile = args[0];
		String outputFile = args[1];
		int verbThreshold = Integer.parseInt(args[2]);
		int objThreshold = Integer.parseInt(args[3]);
		
		BufferedReader in = new BufferedReader(new FileReader(inputFile));
		String line = in.readLine();
		
		// Collect verb obj counts
		Map<String, Integer> verbToCount = new HashMap<String, Integer>();
		Map<String, Integer> objToCount = new HashMap<String, Integer>();
		int totalInputRow = 0;
		
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = stem(toks[1]);
			String obj = stem(toks[2]);
			int freq = Integer.parseInt(toks[3]);
			
			StatUtil.addToTally(verbToCount, verb, freq);
			StatUtil.addToTally(objToCount, obj, freq);
			
			line = in.readLine();
			totalInputRow++;
			if(totalInputRow % 100000 == 0) {
				System.out.printf("Processed %d rows\n", totalInputRow);
			}
		}
		
		// Filter VO pairs based on the VO counts
		PrintWriter out = new PrintWriter(outputFile);
		in.close();
		in = new BufferedReader(new FileReader(inputFile));
		line = in.readLine();
		int totalOutputRow = 0;
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = stem(toks[1]);
			String obj = stem(toks[2]);
			
			int vcount = verbToCount.get(verb);
			int ocount = objToCount.get(obj);
			if(vcount >= verbThreshold && ocount >= objThreshold) {
				out.println(verb + "\t" + obj);
				totalOutputRow++;
			}
			
			line = in.readLine();
		}
		
		System.out.printf("Written %d out of %d rows.\n", totalOutputRow, totalInputRow);
		
		in.close();
		out.close();
	}
	
	/**
	 * @param s
	 * @return stemmed version of 's'
	 */
	private static String stem(String s) {
		for(int i = 0; i < s.length(); i++) {
			stemmer.add(s.charAt(i));
		}
		stemmer.stem();
		return stemmer.toString();
	}
}
