package robertslab.hadoop.io;

import java.io.IOException;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.mapreduce.RecordWriter;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

public class SFileRecordWriter extends RecordWriter<SFileHeader, SFileRecord>
{
	protected FSDataOutputStream out;

	SFileRecordWriter(FSDataOutputStream out)
	{
		this.out = out;
	}
	
	@Override
	public void close(TaskAttemptContext arg0) throws IOException,InterruptedException
	{
		out.close();
	}

	@Override
	public void write(SFileHeader header, SFileRecord record) throws IOException,InterruptedException
	{
		header.write(out);
		record.write(out);
	}
}
