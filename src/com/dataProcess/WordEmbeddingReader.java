package com.dataProcess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

public class WordEmbeddingReader {
	public HashMap<String, Double[]> readWordEmbedding( ){
		ArrayList<String> termList = new ArrayList<String>();
		try {
			InputStreamReader inputStream = new InputStreamReader ( new FileInputStream ( new File (  "data/input/word2vec/termList.txt" ) ), "utf-8"); 
			BufferedReader reader = new BufferedReader ( inputStream );
			String line = null;
			while (  (line = reader.readLine()) != null ){				
				termList.add( line.trim() );
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		ArrayList<Double[]> wordEmbList = new ArrayList<Double[]>();
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File (  "data/input/word2vec/embList.txt" )));
			String line = null;
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(" ");
				
				Double[] values = new Double[temp.length];
				for ( int i =0; i < temp.length; i++ ){
					values[i] = Double.parseDouble( temp[i] );
				}
				wordEmbList.add( values );
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		if ( termList.size() != wordEmbList.size() ){
			System.out.println ( "Unequal word2vec size " + termList.size() + " " + wordEmbList.size() ); 
		}
		
		HashMap<String, Double[]> word2Vec = new HashMap<String, Double[]>();
		for ( int i =0; i < termList.size(); i++ ){
			String term = termList.get(i);
			Double[] values = wordEmbList.get( i );
			
			word2Vec.put( term,  values );			
		}
		return word2Vec;
	}
}
