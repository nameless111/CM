package com.evaluation;

import java.util.HashMap;

public class PerformanceCombine {
	//主要用于不同方法见的性能比较
	public void combinePerformanceDifferentMethods ( String folderName, String[] fileNames, String[] displayMethodNames , String outFile ) {
		HashMap<String, HashMap<Integer, Double[]>> totalPerformance = new HashMap<String, HashMap<Integer, Double[]>>();
		
		PerformanceReader perfReader = new PerformanceReader();
		for ( int i = 0; i < fileNames.length; i++ ){
			String methodName = fileNames[i];
			String fileName = folderName + "/" + methodName;
			System.out.println( fileName );
			
			HashMap<Integer, Double[]> performanceList = perfReader.readPerformance(fileName); 
			totalPerformance.put( methodName, performanceList );
		}
		
		PerformanceWriter perfWriter = new PerformanceWriter ();
		perfWriter.writePerformancePlot( outFile, totalPerformance, fileNames, displayMethodNames);
	}
	
	public static void main ( String[] args ) {
		PerformanceCombine perfComb = new PerformanceCombine();
		/*
		String[] fileNames = {"performance-total-total.csv",  "performance-total-Domain.csv", "performance-total-Naive.csv"};
		String[] displayMethodNames = { "TARO", "InterestDriven", "Naive"};
		*/
		/*
		String[] fileNames = {"performance-total-total.csv",  "performance-total-total-8.csv", "performance-total-total-12.csv", 
				"performance-total-total-18.csv"};
		String[] displayMethodNames = { "0:00", "6:00", "12:00", "18:00"};
		*/
		/*
		String[] fileNames = { "performance-total-total.csv", "performance-total-act.csv", "performance-total-exp.csv", 
		"performance-total-task.csv" };
		String[] displayMethodNames = { "TARO", "act", "exp", "task" };
		perfComb.combinePerformanceDifferentMethods("data/output/weka2/sum" , fileNames, displayMethodNames, "data/output/weka2/sum/compare.csv");
	*/
		String[] fileNames = { "performance-total-time0.csv", "performance-total-time1.csv", "performance-total-time2.csv", 
				"performance-total-time3.csv", "performance-total-time4.csv", "performance-total-time5.csv", 
				"performance-total-time6.csv",  "performance-total-time7.csv", "performance-total-time8.csv", 
				"performance-total-time9.csv",  "performance-total-time10.csv", "performance-total-time11.csv"};
		String[] displayMethodNames = { "t0", "t1", "t2", "t3", 
				"t4", "t5", "t6", 
				"t7", "t8", "t9", 
				"t10", "t11" };
		
		perfComb.combinePerformanceDifferentMethods("data/output/weka2/sum" , fileNames, displayMethodNames, "data/output/weka2/sum/compare-time.csv");
		
	}
}
