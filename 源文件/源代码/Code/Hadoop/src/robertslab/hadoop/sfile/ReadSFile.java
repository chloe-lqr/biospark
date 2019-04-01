package robertslab.hadoop.sfile;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import robertslab.hadoop.io.SFileHeader;
import robertslab.hadoop.io.SFileInputFormat;
import robertslab.hadoop.io.SFileRecord;

import java.io.IOException;

public class ReadSFile {

	public static void main(String[] args) throws Exception
	{
		if (args.length != 2)
		{
			System.err.printf("Usage: ls input_path output_path\n");
			System.exit(-1);
		}
		
		Job job = Job.getInstance();
		job.setJarByClass(ReadSFile.class);
		job.setJobName("(blank) Read SFile");
		job.setInputFormatClass(SFileInputFormat.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.setMapperClass(ListSFileMapper.class);
		System.exit(job.waitForCompletion(true)?0:1);
	}
	
	public static class ListSFileMapper extends Mapper<SFileHeader,SFileRecord,Text,IntWritable>
	{
	    private final static IntWritable one = new IntWritable(1);
	    private Text word = new Text();
	    
		public void map(SFileHeader key, SFileRecord value, Context context)
		throws IOException, InterruptedException
		{
			word.set(key.getName());
	        context.write(word, one);
		}
	}
}
