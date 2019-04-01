package robertslab.hadoop.sfile;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import robertslab.hadoop.io.SFileHeader;
import robertslab.hadoop.io.SFileInputFormat;
import robertslab.hadoop.io.SFileOutputFormat;
import robertslab.hadoop.io.SFileRecord;

public class SFileStream
{

	public static void main(String[] args) throws Exception
	{
		String inputPath=args[0];
		String outputPath=args[1];
		String mapperBin="/bin/cat";
		String reducerBin="";
		
		// Parse the arguments.
		if (false)
		{
			System.err.printf("Usage: -input input_path -output output_path -mapper mapper_bin -reducer reducer_bin	\n");
			System.exit(-1);
		}
		
		Job job = Job.getInstance();
		Configuration conf = job.getConfiguration();
		job.setJarByClass(SFileStream.class);
		job.setJobName("SFile Stream:"+mapperBin+"/"+reducerBin);
		job.setInputFormatClass(SFileInputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		job.setOutputFormatClass(SFileOutputFormat.class);
		job.setOutputKeyClass(SFileHeader.class);
		job.setOutputValueClass(SFileRecord.class);
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.setMapperClass(SFileStreamMapper.class);
		conf.set("sfile_mapper_executable", mapperBin);
		if (reducerBin != "")
		{
			job.setReducerClass(SFileStreamReducer.class);
			conf.set("sfile_reducer_executable", reducerBin);
		}
		else
		{
			job.setNumReduceTasks(0);
		}
		System.exit(job.waitForCompletion(true)?0:1);
	}
	
	public static class SFileStreamMapper extends Mapper<SFileHeader,SFileRecord,SFileHeader,SFileRecord>
	{
    	Process mapperProcess=null;
    	DataOutputStream processInput;
    	DataInputStream processOutput;
	    
	    protected void setup(org.apache.hadoop.mapreduce.Mapper<SFileHeader,SFileRecord,SFileHeader,SFileRecord>.Context context)
	    throws IOException, InterruptedException
	    {
	    	Configuration conf = context.getConfiguration();
	    	String executable = conf.get("sfile_mapper_executable");
	    	System.out.printf("mapper setup: %s\n",executable);
	    	mapperProcess = Runtime.getRuntime().exec(executable);
	    	processInput = new DataOutputStream(mapperProcess.getOutputStream());
	    	processOutput = new DataInputStream(mapperProcess.getInputStream());
        }
	    
		protected void map(SFileHeader key, SFileRecord value, org.apache.hadoop.mapreduce.Mapper<SFileHeader,SFileRecord,SFileHeader,SFileRecord>.Context context)
		throws IOException, InterruptedException
		{
			final SFileHeader outputkey = new SFileHeader();
			final SFileRecord outputvalue = new SFileRecord();
			
			// Start a thread to read the values.
			Thread outputReaderThread = new Thread()
			{
			    public void run() {
			    	try
			    	{
						outputkey.readFields(processOutput);
						outputvalue.readFields(processOutput);
			    	}
			    	catch(IOException e)
			    	{
						e.printStackTrace();			    		
			    	}
			    }
			};
			outputReaderThread.start();
			
			// Write the values.
			key.write(processInput);
			value.write(processInput);
			processInput.flush();
			
			// Wait until the reader finishes.
			outputReaderThread.join();
			
			// Write the output to the context.
	        context.write(outputkey, outputvalue);

		}
	    
	    protected void cleanup(org.apache.hadoop.mapreduce.Mapper<SFileHeader,SFileRecord,SFileHeader,SFileRecord>.Context context)
        throws IOException, InterruptedException
        {
	    	System.out.printf("mapper cleanup\n");
	    	processInput.flush();
	    	processInput.close();
	    	processOutput.close();
	    	boolean exitted=false;
	    	for (int i=0; i<5 && !exitted; i++)
	    	{
		    	try
		    	{
		    		int exitValue = mapperProcess.exitValue();
		    		exitted = true;
		    		System.out.printf("mapper exitted with value %d\n",exitValue);
		    	}
		    	catch (IllegalThreadStateException e)
		    	{
		    		Thread.sleep(1000);
		    	}
	    	}
	    	if (!exitted)
	    	{
	    		mapperProcess.destroy();
	    		System.out.printf("killed mapper\n");
	    	}
        }
	}

	public static class SFileStreamReducer extends Reducer<SFileHeader,SFileRecord,SFileHeader,SFileRecord>
	{
		String executable;
		
	    protected void setup(org.apache.hadoop.mapreduce.Reducer<SFileHeader,SFileRecord,SFileHeader,SFileRecord>.Context context)
	    throws IOException, InterruptedException
	    {
	    	Configuration conf = context.getConfiguration();
	    	executable = conf.get("sfile_reducer_executable");
	    	System.out.printf("reducer setup: %s\n",executable);
        }
	    
		public void reduce(SFileHeader key, Iterable<SFileRecord> values, Context context)
		throws IOException, InterruptedException
		{
		      for (SFileRecord value : values) {
			      context.write(key, value);
		      }
		}
	    
	    protected void cleanup(org.apache.hadoop.mapreduce.Reducer<SFileHeader,SFileRecord,SFileHeader,SFileRecord>.Context context)
        throws IOException, InterruptedException
        {
	    	System.out.printf("reducer cleanup: %s\n",executable);
        }
	}

}
