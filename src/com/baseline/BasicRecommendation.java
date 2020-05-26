package com.baseline;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;

import com.data.TestProject;
import com.taskRecommendation.RecommendationTime;

public class BasicRecommendation {
	LinkedHashMap<Date, ArrayList<TestProject>> trainProjectList;
	LinkedHashMap<Date, ArrayList<TestProject>> testProjectList;
	Integer testBeginIndex = 50;   
	
	public BasicRecommendation ( ){
		trainProjectList = new LinkedHashMap<Date, ArrayList<TestProject>>();
		testProjectList = new LinkedHashMap<Date, ArrayList<TestProject>>();
	}
	
	public void separateTrainTestSet ( ArrayList<TestProject> projectList ){
		RecommendationTime recTimeTool = new RecommendationTime ();
		LinkedHashMap<Date, ArrayList<TestProject>> recTimeByProjects = recTimeTool.obtainMultiTaskStatus(projectList);
		
		int trainCount = 0;
		for ( Date curTime : recTimeByProjects.keySet() ){
			trainCount++;
			ArrayList<TestProject> curProjectList = recTimeByProjects.get( curTime );
			if ( trainCount < testBeginIndex ){   /////
				trainProjectList.put( curTime, curProjectList );
			}else{
				testProjectList.put( curTime, curProjectList );
			}			
		}
	}
}
