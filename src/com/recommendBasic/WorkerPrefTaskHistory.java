package com.recommendBasic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import com.data.Constants;
import com.data.TestProject;
import com.data.TestReport;
import com.dataProcess.ReportSegment;
import com.dataProcess.TestProjectReader;

public class WorkerPrefTaskHistory {
	
	//全部worker在该数据集上的所有参加任务的时间和任务描述；当在某个时间点进行推荐时，该时间点后面的活动自动忽略，只选取该时间点之前的活动作为该人员的经验
	//该类和WorkerPreferenceHistory的区别在于，该类是对于参加的任务进行统计，preference是对于在任务中提交的报告进行统计
	public HashMap<String, HashMap<Date, ArrayList<List<String>>>> retrieveWorkerPreferenceHistory ( ArrayList<TestProject> projectList ){
		HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerPreferenceHistory = new HashMap<String, HashMap<Date, ArrayList<List<String>>>>();
		//<worker, <Date, >>, List<String> is the terms from the task requirement
		ReportSegment segTool = new ReportSegment();
		
		for ( int i =0; i < projectList.size(); i++ ){
			TestProject project = projectList.get( i );
			ArrayList<TestReport> reportList = project.getTestReportsInProj();
			Date projTime = reportList.get(0).getSubmitTime();
			
			HashSet<String> workers = new HashSet<String>();
			for ( int j =0; j < reportList.size(); j++ ) {
				workers.add( reportList.get(j).getUserId() );
			}
			
			List<String> termList = project.getTestTask().getTaskDescription();
			for ( String workerId : workers ) {
				if ( workerPreferenceHistory.containsKey( workerId )){
					HashMap<Date, ArrayList<List<String>>> history = workerPreferenceHistory.get( workerId );
					if ( history.containsKey( projTime )){
						ArrayList<List<String>> reportsList = history.get( projTime );
						reportsList.add( termList );
						history.put( projTime, reportsList);
					}else{
						ArrayList<List<String>> reportsList = new ArrayList<List<String>>();
						reportsList.add( termList );
						history.put( projTime, reportsList);
					}
					workerPreferenceHistory.put( workerId, history );
				}
				else{
					HashMap<Date, ArrayList<List<String>>> history = new HashMap<Date, ArrayList<List<String>>>();
					ArrayList<List<String>> reportsList = new ArrayList<List<String>>();
					reportsList.add( termList );
					history.put( projTime, reportsList);
					
					workerPreferenceHistory.put( workerId, history );
				}
			}
		}
		return workerPreferenceHistory;
	}
	
	public void storeWorkerPreferenceHistory ( HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerPreferenceHistory, String fileName ){ 
		WorkerExpertiseHistory historyTool = new WorkerExpertiseHistory ();
		historyTool.storeWorkerExpertiseHistory(workerPreferenceHistory, fileName);
	}
	
	public HashMap<String, HashMap<Date, ArrayList<List<String>>>> readWorkerPreferenceHistory ( String fileName ){
		WorkerExpertiseHistory historyTool = new WorkerExpertiseHistory ();
		return historyTool.readWorkerExpertiseHistory(fileName);
	}
	
	public HashMap<String, HashMap<Date, Double[]>> retrieveWorkerPreferenceHistorySemantic ( HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerPreferenceHistory ){
		WorkerExpertiseHistory historyTool = new WorkerExpertiseHistory ();
		return historyTool.retrieveWorkerExpertiseHistorySemantic(workerPreferenceHistory);
	}
	public void storeWorkerPreferenceHistorySemantic ( HashMap<String, HashMap<Date, Double[]>> semanticWorkerPreferenceHistory, String fileName ){ 
		WorkerExpertiseHistory historyTool = new WorkerExpertiseHistory ();
		historyTool.storeWorkerExpertiseHistorySemantic(semanticWorkerPreferenceHistory, fileName);
	}
	public HashMap<String, HashMap<Date, Double[]>> readWorkerPreferenceHistorySemantic ( String fileName ){
		WorkerExpertiseHistory historyTool = new WorkerExpertiseHistory ();
		return historyTool.readWorkerExpertiseHistorySemantic(fileName);
	}
	
	public static void main ( String args[] ){
		WorkerPrefTaskHistory history = new WorkerPrefTaskHistory();
		
		TestProjectReader projectReader = new TestProjectReader();
		ArrayList<TestProject> projectList = projectReader.loadTestProjectAndTaskList(Constants.PROJECT_FOLDER, Constants.TASK_DES_FOLDER );
 		HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerPreferenceHistory = history.retrieveWorkerPreferenceHistory(projectList); 
		
 		history.storeWorkerPreferenceHistory(workerPreferenceHistory, "data/output/history/preference.txt" );
		
		/*
 		HashMap<String, HashMap<Date, ArrayList<List<String>>>> storedHistory = history.readWorkerExpertiseHistory( "data/output/history/preference.txt");
		
		HashMap<Date, ArrayList<List<String>>> historyInfo = storedHistory.get( "14471438" );
		for ( Date date : historyInfo.keySet() ){
			System.out.println( Constants.dateFormat.format( date ) + " " + historyInfo.get(date).size() );
			ArrayList<List<String>> info = historyInfo.get( date );
			for ( int i =0; i < info.size(); i++ ){
				for ( int j =0; j < info.get(i).size(); j++ ){
					System.out.print( info.get(i).get(j) + "*");
				}
			}
			System.out.println ();					
		}
		*/
	}
}
