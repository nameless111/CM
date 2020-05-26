package com.taskRecommendation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class TaskRecommendationAndPerformance {
	//基于predictWorkerPerformance的结果，进行task recommendation
	//格式为<project-workerId, trueLabel, predictLabel, predictProb>
	public HashMap<String, ArrayList<String>> recommendTaskStatus ( HashMap<String, String[]> predictDetailList, boolean isPredicted ){
		HashMap<String, ArrayList<String>> tasksRecForWorkers = new HashMap<String, ArrayList<String>>();   //<workerId, List<ProjectName>>
		
		for ( String key : predictDetailList.keySet() ){
			//System.out.println (key);
			String[] temp = key.split("----");
			String projectName = temp[0].trim();
			String workerId = temp[1].trim();
			
			ArrayList<String> tasksRec = new ArrayList<String>();
			String[] predictDetail = predictDetailList.get( key);
			int index = 0;
			if ( isPredicted ){    //表示查找的是预测状态下的推荐任务；否则为真实情况下的推荐任务
				index = 1;
			}
			if ( predictDetail[index].equals("yes")){
				if ( tasksRecForWorkers.containsKey( workerId )){
					tasksRec = tasksRecForWorkers.get( workerId );
				}
				tasksRec.add( projectName );
			}
			tasksRecForWorkers.put( workerId, tasksRec );
		}
		return tasksRecForWorkers;
	}
	
	public void computeRecommendationPrecisionRecall ( HashMap<String, String[]> predictDetailList, String outFile ){
		RecommendationTime recTimeTool = new RecommendationTime (); 
		HashMap<Integer, String[]> recTimeStatus = recTimeTool.readMultiTasks();
		
		int beginindex = outFile.lastIndexOf( "-");
		int endIndex = outFile.indexOf(".csv");
		String subStr = outFile.substring( beginindex+1, endIndex);
		Integer index = Integer.parseInt( subStr );
		
		String multiTaskNumStr = recTimeStatus.get( index)[1];
		Integer multiTaskNum = Integer.parseInt( multiTaskNumStr );
		
		HashMap<String, ArrayList<String>> recTasksList = this.recommendTaskStatus(predictDetailList, true );
		HashMap<String, ArrayList<String>> trueTasksList = this.recommendTaskStatus(predictDetailList, false);
		
		//对于每个worker，都有precision and recall
		HashMap<String, Double[]> performanceList = new HashMap<String, Double[]>();
		for ( String workerId : recTasksList.keySet() ){
			ArrayList<String> predictTasks = recTasksList.get( workerId );
			ArrayList<String> trueTasks = trueTasksList.get( workerId );
			
			int truePositive = 0;
			for ( int i =0; i < predictTasks.size(); i++ ){
				String task = predictTasks.get(i);
				if ( trueTasks.contains( task )){
					truePositive++;
				}
			}
			Double precision = 0.0;
			if ( predictTasks.size() > 0 )
				precision = 1.0*truePositive / predictTasks.size();
			Double recall = 0.0;
			if ( trueTasks.size() > 0 )
				recall = 1.0*truePositive / trueTasks.size();
			Double FMeasure = 0.0;
			if ( precision > 0.0 || recall > 0.0 )
				FMeasure = 2*precision*recall / (precision+recall);
			
			Double[] performance = { precision, recall, FMeasure};
			
			performanceList.put( workerId, performance );
		}
		
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( outFile ) ));
			/*
			for (String workerId : recTasksList.keySet() ){
				writer.write( workerId + ",");
				for ( int i =0; i < recTasksList.get(workerId).size(); i++ ){
					writer.write( recTasksList.get( workerId).get(i) + ",");
				}
				writer.write( " " +",");
				for ( int i =0; i < trueTasksList.get(workerId).size(); i++ ){
					writer.write( trueTasksList.get( workerId).get(i) + ",");
				}
				writer.newLine();
				System.out.println ( );
			}
			*/
			
			for ( String workerId : performanceList.keySet() ){
				Double[] performance = performanceList.get( workerId );
				writer.write( workerId + "," + multiTaskNum + "," + recTasksList.get(workerId).size() + "," + trueTasksList.get(workerId).size() + ",");
				for ( int i =0; i < performance.length; i++ ){
					writer.write( performance[i] + ",");
				}
				writer.newLine();
			}
			
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public HashMap<String, String[]> readPredictDetails (String fileName ){
		HashMap<String, String[]> predictDetailList = new HashMap<String, String[]>();
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File ( fileName )));
			String line = null;
			line = reader.readLine();
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(",");
				String key = temp[1];
				String[] values = { temp[2], temp[3], temp[4] };
				
				predictDetailList.put( key, values );
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return predictDetailList;
	}
	
	public static void main ( String[] args ){
		TaskRecommendationAndPerformance recPerformance = new TaskRecommendationAndPerformance();
		int beginIndex = 50, endIndex = 142;
		for ( int i = beginIndex; i <= endIndex; i++ ){
			HashMap<String, String[]> predictDetailList = recPerformance.readPredictDetails( "data/output/weka/result/result-" + i + ".csv");
			recPerformance.computeRecommendationPrecisionRecall(predictDetailList, "data/output/weka/performance/performance-" + i + ".csv");
		}
	}
}
