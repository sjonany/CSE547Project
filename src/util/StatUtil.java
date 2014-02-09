package util;

import java.util.Map;

public class StatUtil {
	public static <K> void addToTally(Map<K, Integer> tally, K key, int val) {
		if(!tally.containsKey(key)) {
			tally.put(key, val);
		} else {
			tally.put(key, tally.get(key) + val);
		}
	}
}
