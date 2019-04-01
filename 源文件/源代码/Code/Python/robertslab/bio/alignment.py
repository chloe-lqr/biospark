from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def calculateConsensus(alignment, gapFraction=0.3):
    
    consensus = ""
    for col in range(0, alignment.get_alignment_length()):
        gaps=0
        counts={}
        for row in range(0, len(alignment)):
            if alignment[row,col] == '-':
                gaps+=1
            elif alignment[row,col] in counts:
                counts[alignment[row,col]]+=1
            else:
                counts[alignment[row,col]]=1
                
        if gaps > int(float(len(alignment))*gapFraction):
            consensus += "-"
        else:
            maxCount=0
            maxValue=""
            for value in counts.iterkeys():
                if counts[value] > maxCount:
                    maxCount=counts[value]
                    maxValue=value
            if maxCount > 0:
                consensus += maxValue
            else:
                consensus += "X"
    return SeqRecord(Seq(consensus, alignment._alphabet), id="consensus")
