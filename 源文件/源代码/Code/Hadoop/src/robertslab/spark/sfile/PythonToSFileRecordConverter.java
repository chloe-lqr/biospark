package robertslab.spark.sfile;

import org.apache.spark.api.python.Converter;

import robertslab.hadoop.io.SFileRecord;

@SuppressWarnings("serial")
public class PythonToSFileRecordConverter<T, U> implements Converter<T, U> {
	
	@SuppressWarnings("unchecked")
	@Override
	public U convert(T arg)
	{
		System.out.println("Converting record of type"+arg.getClass().getName());
		if (arg instanceof byte[])
		{
			return (U)new SFileRecord((byte[])arg);
		}
	    return null;
	}

}
