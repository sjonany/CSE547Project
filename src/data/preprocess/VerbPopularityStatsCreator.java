package data.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import util.StatUtil;

/**
 * Should be run on the training set.
 * Create a text file where each row is
 * <verb> <number of training instances>
 * The rows are sorted in descending order of #training instances
 */
public class VerbPopularityStatsCreator {
	public static void main(String[] args) throws Exception {
		if(args.length != 2) {
			System.err.println("Usage: <train file> <output file>");
			return;
		}
		
		// format of input file - see SvmlightTrainingDriver
		String inputFile = args[0];
		String outputFile = args[1];
		
		BufferedReader in = new BufferedReader(new FileReader(inputFile));

		final Map<String, Integer> verbToClusterSize = new HashMap<String, Integer>();
		String line = in.readLine();
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = toks[0];
			StatUtil.addToTally(verbToClusterSize, verb, 1);
			line = in.readLine();
		}
		
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
		
		PrintWriter out = new PrintWriter(outputFile);
		for(Entry<String, Integer> entry : entries) {
			out.println(entry.getKey() + "\t" + entry.getValue());
		}
		
		in.close();
		out.close();
	}
}
