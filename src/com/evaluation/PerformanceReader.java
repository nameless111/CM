package com.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class PerformanceReader {
	public ArrayList<ArrayList<Double>> readRecommendationPerformance ( String fileName ){
		//workerId, multiTaskNum, recTaskNum, trueTaskNum, precision, recall, FMeasure		
		ArrayList<Double> precisionList = new ArrayList<Double>();
		ArrayList<Double> recallList = new ArrayList<Double>();
		//ArrayList<Double> FMeasureList = new ArrayList<Double>();
		ArrayList<Double> saveEffortList = new ArrayList<Double>();
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File ( fileName )));
			String line = null;
			line = reader.readLine();
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(",");
				Integer multiTasks = Integer.parseInt( temp[1]);
				Integer recTasks = Integer.parseInt( temp[2] );
				Integer trueTasks = Integer.parseInt( temp[3] );
				
				if ( trueTasks > 0 ){
					Double precision = Double.parseDouble( temp[4] );
					Double recall = Double.parseDouble( temp[5] );
					//Double FMeasure = Double.parseDouble( temp[6] );
					
					precisionList.add( precision );
					recallList.add( recall );
					//FMeasureList.add ( FMeasure );
					
					Double saveEffort = (multiTasks - recTasks )/ (1.0* multiTasks );
					saveEffortList.add( saveEffort );
				}
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		ArrayList<ArrayList<Double>> performanceDetailList = new ArrayList<ArrayList<Double>>();
		performanceDetailList.add( precisionList );
		performanceDetailList.add( recallList );
		//performanceDetailList.add( FMeasureList );
		performanceDetailList.add( saveEffortList );
		
		return performanceDetailList;
	}
	
	//Id, precision, recall, savedEffort
	public HashMap<Integer, Double[]> readPerformance ( String fileName ) {
		HashMap<Integer, Double[]> attrValues = new HashMap<Integer, Double[]>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader( new File ( fileName )));	
			String line = "";
			br.readLine();
			while ( (line = br.readLine()) != null ){
				String[] temp = line.split( ",");
				String[] temp2 = temp[0].split("-");
				String index = temp2[1].replace( ".csv", "" );
				Integer projectId = Integer.parseInt( index );
				
				Double precison = Double.parseDouble( temp[1]);
				Double recall = Double.parseDouble( temp[2] );
				Double savedEffort = Double.parseDouble( temp[3] );
				
				Double[] values = { precison, recall, savedEffort};
				attrValues.put( projectId, values);
			}
			br.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return attrValues;
	}
	
}
