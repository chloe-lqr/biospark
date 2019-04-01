package robertslab.spark.sfile;

import org.apache.spark.api.python.Converter;

import robertslab.hadoop.io.SFileRecord;

@SuppressWarnings("serial")
public class SFileRecordToPythonConverter<T, U> implements Converter<T, U> {
	
	@SuppressWarnings("unchecked")
	@Override
	public U convert(T arg)
	{
		if (arg instanceof SFileRecord)
		{
			SFileRecord record = (SFileRecord)arg;
			return (U)record.getData();
		}
	    return null;
	}

}
