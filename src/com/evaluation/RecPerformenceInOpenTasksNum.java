package com.evaluation;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;

import com.taskRecommendation.RecommendationTime;


public class RecPerformenceInOpenTasksNum {
	public void obtainPerformanceForDifferentOpenTasksNum ( String performanceFile, String outFile, String outPlotFile ) {
		PerformanceReader performanceReader = new PerformanceReader();
		
		RecommendationTime recTimeTool = new RecommendationTime();
		HashMap<Integer, String[]> recTimeTasks = recTimeTool.readMultiTasks();
		
		HashMap<Integer, Double[]> performance = performanceReader.readPerformance( performanceFile );
		
	}
	
	public static void main ( String[] args ){
		RecPerformenceInOpenTasksNum performanceTool = new RecPerformenceInOpenTasksNum();
		
	}
}
