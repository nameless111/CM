package com.taskRecommendation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import com.data.TestProject;
import com.data.TestReport;

public class PositiveNegativeForFeaturePreparation {
	//目前是没有考虑重复缺陷的情况，就是发现了缺陷，就认为是positive
	public ArrayList<String> retrievePredictionLabel ( TestProject project, int recTimePoint ){
		ArrayList<TestReport> reportList = project.getTestReportsInProj();
		
		ArrayList<String> bugWorkerList = new ArrayList<String>();
		for ( int i = recTimePoint+1; i < reportList.size(); i++ ){
			TestReport report = reportList.get( i );
			String bugTag = report.getTag();
			if ( bugTag.equals( "审核通过") && !bugWorkerList.contains( report.getUserId()) ){
				bugWorkerList.add( report.getUserId() );
			}
		}
		
		return bugWorkerList;
	}
	
	//随机选择两倍于positiveWorkerList的workers
	public ArrayList<String> retrieveNegativeSampleTrainset ( HashMap<String, ArrayList<Double>> totalFeatureList, ArrayList<String> positiveWorkerList ){
		int negSize = positiveWorkerList.size() * 2;
		ArrayList<String> negativeWorkerList = new ArrayList<String>();
		
		ArrayList<String> allCandWorkers = new ArrayList<String>();
		for ( String worker : totalFeatureList.keySet() ){
			allCandWorkers.add( worker );
		}
		
		Random rand = new Random();
		for ( int i =0; i < negSize; ){
			int index = rand.nextInt( allCandWorkers.size() );
			String worker = allCandWorkers.get( index );
			if ( positiveWorkerList.contains( worker ) || negativeWorkerList.contains( worker )){
				continue;
			}
			
			negativeWorkerList.add( worker );
			i++;
		}
		return negativeWorkerList;
	}
	
	//目前只是在com.exploration中，随机选择正样本
	public ArrayList<String> retrieveNegativeSampleTrainset ( ArrayList<String> candWorkerList, ArrayList<String> positiveWorkerList  ){
		int negSize = positiveWorkerList.size();
		ArrayList<String> negativeWorkerList = new ArrayList<String>();
		
		Random rand = new Random();
		for ( int i =0; i < negSize; ){
			int index = rand.nextInt( candWorkerList.size() );
			String worker = candWorkerList.get( index );
			if ( positiveWorkerList.contains( worker ) || negativeWorkerList.contains( worker )){
				continue;
			}
			
			negativeWorkerList.add( worker );
			i++;
		}
		return negativeWorkerList;
	}
	
	//trainset和testset区别在于，testset是所有的negative sample
	public ArrayList<String> retrieveNegativeSampleTestset ( HashMap<String, ArrayList<Double>> totalFeatureList, ArrayList<String> positiveWorkerList ){
		ArrayList<String> negativeWorkerList = new ArrayList<String>();
		
		for ( String worker : totalFeatureList.keySet() ){
			if ( !positiveWorkerList.contains( worker ) && !negativeWorkerList.contains( worker )){
				negativeWorkerList.add( worker );
			}			
		}
		return negativeWorkerList;
	}
}
