package com.dataProcess;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import com.data.TestProject;
import com.data.TestReport;



public class InverseDocFrequency {

	public InverseDocFrequency() {
		// TODO Auto-generated constructor stub
	}
	
	public HashMap<String, Integer> countDocumentFrequencyTotal ( ArrayList<HashMap<String, Integer>> totalDataSet ){
		HashMap<String, Integer> documentFrequency = new HashMap<String, Integer>();
		
		HashSet<String> representWord = new HashSet<String>();
		for ( int i =0; i < totalDataSet.size(); i++ ) {
			HashMap<String, Integer> dataInstance = totalDataSet.get( i);
			
			representWord.addAll( dataInstance.keySet() );
		}
		
		for ( String key : representWord ) {
			//search how many dataInstance this key has appeared
			for ( int i =0; i < totalDataSet.size(); i++ ) {
				HashMap<String, Integer> dataInstance = totalDataSet.get( i );
				if ( dataInstance.containsKey( key )) {
					int num = 1;
					if ( documentFrequency.containsKey( key )) {
						num = documentFrequency.get( key ) + 1;
					}
					documentFrequency.put( key, num );
				}
			}
		}
		
		return documentFrequency;
	}
	
	//obtain the inverse document frequency based on the documentFrequency and documentNum 
	public HashMap<String, Double> getInverseDocumentFrequency( ArrayList<HashMap<String, Integer>> totalDataSet ){
		HashMap<String, Integer> documentFrequency = this.countDocumentFrequencyTotal(totalDataSet);
		int documentNum = totalDataSet.size();
		
		HashMap<String, Double> ivDocumentFrequency = new HashMap<String, Double>();
		
		Iterator iter = documentFrequency.entrySet().iterator();
		while ( iter.hasNext() ){
			Map.Entry<String, Integer> entry = (Entry<String, Integer>) iter.next();
			String word = entry.getKey();
			Double frequency = entry.getValue().doubleValue();
			
			frequency = (1.0* documentNum ) / (1.0*frequency);
			frequency = Math.log( frequency );
			
			ivDocumentFrequency.put( word , frequency );
		}		
		
		return ivDocumentFrequency;
	}
}
