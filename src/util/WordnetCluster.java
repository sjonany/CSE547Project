package util;
import edu.smu.tspell.wordnet.Synset;
import edu.smu.tspell.wordnet.SynsetType;
import edu.smu.tspell.wordnet.WordNetDatabase;
import edu.smu.tspell.wordnet.impl.file.ReferenceSynset;

public class WordnetCluster {
	// first number of the noun Lexicographer Files
	// http://wordnet.princeton.edu/wordnet/man/lexnames.5WN.html
	private static final int NOUN_CLUS_NUM_OFFSET = 3;
	
	private String dictPath;
	private WordNetDatabase database;
	
	public WordnetCluster (String dictPath) {
		this.dictPath = dictPath;
		System.setProperty("wordnet.database.dir", this.dictPath);
		
		this.database = WordNetDatabase.getFileInstance();
	}

	// 26-bit length, a 1 on the ith index means the noun belongs to that cluster
	public int getClusterVector (String unstemmedNoun) {
		int result = 0;
		
		// Get the synsets containing the wrod form		
		Synset[] synsets = this.database.getSynsets(unstemmedNoun);
		
		if (synsets.length > 0)
		{
			for (int i = 0; i < synsets.length; i++)
			{
				if(synsets[i].getType() == SynsetType.NOUN) {
					ReferenceSynset rs = (ReferenceSynset) synsets[i];
					int nounClusID = rs.getLexicalFileNumber() - NOUN_CLUS_NUM_OFFSET;
					result |= (1 << nounClusID);
				}
			}
		}
		
		return result;
	}
	
	public static void main(String[] args) {
		//WordnetCluster WnC = new WordnetCluster("/homes/iws/shaoc/Downloads/WordNet-3.0/dict");
		//WnC.getClusterVector("fly");
		//WnC.getClusterVector("pizza");
	}
}
