package com.evaluation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;

import com.data.Constants;

public class PerformanceWriter {
	String[] category = { "precision", "recall", "savedEffort"};
	
	public void writePerformance (String outFile, LinkedHashMap<String, ArrayList<ArrayList<Double>>> performanceList ) {
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( outFile) ));
			writer.write(" " + ",");
			for ( int i =0; i < category.length; i++ ){
				writer.write( category[i] + ",");
			}
			writer.newLine();
			for ( String recGroup : performanceList.keySet() ){
				ArrayList<ArrayList<Double>> performanceDetailList = performanceList.get( recGroup );
				ArrayList<Double> precisionList = performanceDetailList.get(0);
				ArrayList<Double> recallList = performanceDetailList.get(1);
				//ArrayList<Double> FMeasureList = performanceDetailList.get(2);
				ArrayList<Double> saveEffortList = performanceDetailList.get(2);
				for ( int i =0; i < precisionList.size(); i++ ){
					writer.write( recGroup + "," + precisionList.get(i) +"," + recallList.get(i) +"," + saveEffortList.get(i));
					writer.newLine();
				}
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void writePerformancePlot (String outFile, LinkedHashMap<String, ArrayList<ArrayList<Double>>> performanceList ) {
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( outFile ) ));
			writer.write(" " + "," + "  ");
			writer.newLine();
			for ( String recGroup : performanceList.keySet() ){
				ArrayList<ArrayList<Double>> performanceDetailList = performanceList.get( recGroup );
				for ( int i =0; i < 3; i++ ) {
					ArrayList<Double> valueList = performanceDetailList.get( i);
					for ( int j =0; j < valueList.size(); j++ ){
						writer.write( category[i] + "," + valueList.get(j) );
						writer.newLine();
					}
				}
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void writePerformancePlot (String outFile, HashMap<String, HashMap<Integer, Double[]>> totalPerformance , 
			String[] methodNames, String[] displayMethodNames) {
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( outFile )));
			writer.write( " " + "," + "  " + "," + "performance");
			writer.newLine();
			for ( int k= 0; k < methodNames.length; k++ ){
				String methodName = methodNames[k];
				HashMap<Integer, Double[]> performance = totalPerformance.get( methodName );
				for ( Integer projectId: performance.keySet() ){
					Double[] values = performance.get( projectId );
					for ( int i =0; i < values.length; i++ ){
						writer.write( displayMethodNames[k] + "," + category[i] + "," + values[i]);
						writer.newLine();
					}
				}				
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
