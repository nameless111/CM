package com.taskRecommendation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;

import com.data.Constants;
import com.data.TestProject;
import com.data.TestReport;
import com.dataProcess.TestProjectReader;

public class RecommendationTime {
	public LinkedHashMap<Date, ArrayList<TestProject>> obtainMultiTaskStatus ( ArrayList<TestProject> projectList ){
		SimpleDateFormat formatLine = new SimpleDateFormat ("yyyy/MM/dd HH:mm");
		Date beginTime = null, endTime = null;
		try {
			beginTime = formatLine.parse( "2015/01/05 00:01");   //0:01, 8:01, 12:01, 18:01
			endTime = formatLine.parse( "2016/08/08 08:00");
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		LinkedHashMap<Date, ArrayList<TestProject>> projectListByDate = new LinkedHashMap<Date, ArrayList<TestProject>>();
		Date curTime = beginTime;
		int count = 0;
		Boolean flag = false;
		while ( curTime.before( endTime) ){
			ArrayList<TestProject> projectsByDate = new ArrayList<TestProject>();
			for ( int i =0; i < projectList.size(); i++ ){
				TestProject project = projectList.get( i );
				ArrayList<TestReport> reportList = project.getTestReportsInProj();
				Date thisBegin = reportList.get(0).getSubmitTime();
				Date thisEnd = reportList.get( reportList.size()-1).getSubmitTime();
				
				if ( curTime.after( thisBegin) && curTime.before( thisEnd )){
					projectsByDate.add( project );
				}				
			}
			
			if ( projectsByDate.size() >= 4 ){
				projectListByDate.put( curTime, projectsByDate );
				count++;
			}			
			
			Calendar cad = Calendar.getInstance();
			cad.setTime( curTime );
			cad.add( Calendar.HOUR_OF_DAY, 24);
			curTime = cad.getTime();
			
			flag = false;
			//System.out.println ( curTime );
		}
		System.out.println ( "number of data instances " + count );
		
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( "data/output/findings/multiTasks.csv" ) ));
			int index = 1;
			for ( Date time : projectListByDate.keySet() ){
				writer.write( index + "," + time + "," + projectListByDate.get(time).size() );
				writer.newLine();
				
				index++;
			}
			writer.flush();
			writer.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return projectListByDate;
	}
	
	public HashMap<Integer, String[]> readMultiTasks ( ){
		HashMap<Integer, String[]> multiTaskList = new HashMap<Integer, String[]>();
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File (  "data/output/findings/multiTasks.csv" )));
			String line = null;
			line = reader.readLine();
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(",");
				Integer key = Integer.parseInt( temp[0]  );
				String[] value = { temp[1], temp[2] };
				multiTaskList.put( key, value );
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return multiTaskList;
	}
	
	public static void main ( String[] args ){
		RecommendationTime recTime = new RecommendationTime ();
		
		TestProjectReader projectReader = new TestProjectReader();
		ArrayList<TestProject> projectList = projectReader.loadTestProjectAndTaskList( Constants.PROJECT_FOLDER, Constants.TASK_DES_FOLDER );
		
		recTime.obtainMultiTaskStatus( projectList );
	}
}
