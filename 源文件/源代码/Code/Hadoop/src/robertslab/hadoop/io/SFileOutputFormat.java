/**
 * 
 */
package robertslab.hadoop.io;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * @author eroberts
 *
 */
public class SFileOutputFormat extends FileOutputFormat<SFileHeader, SFileRecord> {

	@Override
	public RecordWriter<SFileHeader, SFileRecord> getRecordWriter(TaskAttemptContext context) throws IOException, InterruptedException
	{
	    Configuration conf = context.getConfiguration();
	    Path file = getDefaultWorkFile(context, ".sfile");
	    FileSystem fs = file.getFileSystem(conf);
	    FSDataOutputStream out = fs.create(file, false);
	    return new SFileRecordWriter(out);
	}

}
