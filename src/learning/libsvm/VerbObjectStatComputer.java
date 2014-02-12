package learning.libsvm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.commons.lang3.tuple.Pair;

import util.StatUtil;

/**
 * Precompute VO-related stats on the provided dataset
 * TODO: Sigh, I realized this is wrong. We should compute these stats before we do the MI filter.
 */
public class VerbObjectStatComputer {
	private Map<String, Integer> verbCount;
	private Map<String, Integer> objCount;
	private Map<Pair<String, String>, Integer> voCount;
	
	// mapping of distinct verb and obj's to distinct 1-based nums 
	// the strings are ordered lexicographically, and so are the id's
	// so the lexi-smallest string has id 1
	private Map<String, Integer> verbToId;
	private Map<String, Integer> objToId;
	private String[] idToVerb;
	private String[] idToObj;
	
	public VerbObjectStatComputer() {
		reset();
	}
	
	public void reset() {
		this.verbCount = new HashMap<String, Integer>();
		this.objCount = new HashMap<String, Integer>();
		this.voCount = new HashMap<Pair<String, String>, Integer>();
		this.verbToId = new HashMap<String, Integer>();
		this.objToId = new HashMap<String, Integer>();
	}
	
	/**
	 * @return -1 if no such verb has been seen 
	 */
	public int mapVerbToId(String verb) {
		Integer ans = verbToId.get(verb);
		return ans == null ? -1 : ans;
	}	
	
	/**
	 * @return -1 if no such obj has been seen 
	 */
	public int mapObjToId(String obj) {
		Integer ans = objToId.get(obj);
		return ans == null ? -1 : ans;
	}
	
	public String mapIdToVerb(int vid) {
		return idToVerb[vid];
	}
	
	public String mapIdToObj(int nid) {
		return idToObj[nid];
	}
	
	/**
	 * computes statistics from the given dataset path
	 * @param datasetPath - follow the format of data.preprocess.TrainTestValidationCreator
	 * @throws IOException 
	 */
	public void load(String datasetPath) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(datasetPath));
		String line = in.readLine();
		SortedSet<String> verbs = new TreeSet<String>();
		SortedSet<String> objs = new TreeSet<String>();
		int lineCount = 0;
		while(line != null) {
			lineCount++;
			if(lineCount % 100000 == 0) 
				System.out.println(lineCount);
			String[] toks = line.split("\t");
			String verb = toks[0];
			String obj = toks[1];
			int freq = Integer.parseInt(toks[2]);
			boolean isPositive = Integer.parseInt(toks[3]) == 1;
			
			if(!isPositive) {
				// we only want to compute stats over the positive training instances
				line = in.readLine();
				continue;
			}
			
			StatUtil.addToTally(verbCount, verb, freq);
			StatUtil.addToTally(objCount, obj, freq);
			StatUtil.addToTally(voCount, Pair.of(verb, obj), freq);
			
			verbs.add(verb);
			objs.add(obj);
			line = in.readLine();
		}
		in.close();
		
		idToVerb = new String[verbs.size()+1];
		idToObj = new String[objs.size()+1];
		for(String verb : verbs) {
			int id = verbToId.size()+1;
			verbToId.put(verb, id);
			idToVerb[id] = verb;
		}
		for(String obj : objs) {
			int id = objToId.size()+1;
			objToId.put(obj, id);
			idToObj[id] = obj;
		}
	}
	
	public int getCountDistinctVerb() {
		return verbToId.size();
	}
	
	public int getCountDistinctObj() {
		return objToId.size();
	}
	
	/**
	 * @return Pr(n|v), or -1 if noun or verb has never been seen before
	 */
	public double getPrNGivenV(String n, String v) {
		Integer vFreq = verbCount.get(v);
		Integer vnFreq = voCount.get(Pair.of(n, v));
		if(vnFreq == null) {
			vnFreq = 0;
		}
		
		if(vFreq == null) {
			return -1;
		} 
		
		return 1.0 * vnFreq / vFreq; 
	}
	
	public double getPrNGivenV(int nid, int vid) {
		if(nid <= 0) {
			// noun hasn't been seen in positive train set
			return 0;
		}
		return getPrNGivenV(idToObj[nid], idToVerb[vid]);
	}
}
