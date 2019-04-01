/**
 * 
 */
package robertslab.hadoop.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

/**
 * @author eroberts
 *
 */
@SuppressWarnings("serial")
public class SFileRecord implements Writable, java.io.Serializable
{
	protected byte[] data;
	
	public static SFileRecord read(DataInput in) throws IOException
	{
		SFileRecord r = new SFileRecord();
        r.readFields(in);
        return r;
    }
	
	public SFileRecord()
	{
		this.data = new byte[0];
	}
	
	public SFileRecord(byte[] data)
	{
		this.data = data;
	}
	
	public long getRecordLength()
	{
		return 8+data.length;
	}
	
	public long getDataLength()
	{
		return data.length;
	}
	
	public byte[] getData()
	{
		return data;
	}
	
	/* (non-Javadoc)
	 * @see org.apache.hadoop.io.Writable#readFields(java.io.DataInput)
	 */
	@Override
	public void readFields(DataInput in) throws IOException
	{
		// Read the data length.
		byte[] b = new byte[8];
		in.readFully(b);
		int dataLength = ((b[7]&0xff)<<56)+((b[6]&0xff)<<48)+((b[5]&0xff)<<40)+((b[4]&0xff)<<32)+((b[3]&0xff)<<24)+((b[2]&0xff)<<16)+((b[1]&0xff)<<8)+(b[0]&0xff);
		
		// Read the data.
		data = new byte[dataLength];
		in.readFully(data);
	}

	/* (non-Javadoc)
	 * @see org.apache.hadoop.io.Writable#write(java.io.DataOutput)
	 */
	@Override
	public void write(DataOutput out) throws IOException
	{
		// Write the data length.
		long dataLength = data.length;
		out.write((int)(dataLength&0xFF));
		out.write((int)((dataLength>>8)&0xFF));
		out.write((int)((dataLength>>16)&0xFF));
		out.write((int)((dataLength>>24)&0xFF));
		out.write((int)((dataLength>>32)&0xFF));
		out.write((int)((dataLength>>40)&0xFF));
		out.write((int)((dataLength>>48)&0xFF));
		out.write((int)((dataLength>>56)&0xFF));

		// Write the name.
		out.write(data);
	}
}
