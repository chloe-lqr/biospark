try:
    from pyspark import SparkConf, SparkContext
except:
    print
    "No spark libraries found, ensure you are running locally."

import os,stat
import lm.io.Methylation_pb2 as Methylation_pb2
import numpy as np
import re
import robertslab.cellio as cellio
import sys
import pandas as pd

# Global variables.
output_dir = ""

def main_spark(sc, outputFileAddress, inputFileName):

    global output_dir

    # Broadcast the global variables.
    output_dir = sc.broadcast(outputFileAddress)

    records = sc.newAPIHadoopFile(inputFileName,
                                  "robertslab.hadoop.io.SFileInputFormat",
                                  "robertslab.hadoop.io.SFileHeader",
                                  "robertslab.hadoop.io.SFileRecord",
                                  keyConverter="robertslab.spark.sfile.SFileHeaderToPythonConverter",
                                  valueConverter="robertslab.spark.sfile.SFileRecordToPythonConverter")

    # find all methylation related position and put them into one list
    # map: get base information list
    # reduce: combine all reads's base list to ONE list
    findMethylationFirst = records\
        .filter(filterRecordType)\
        .map(extractAlignmentInfo)\
        .reduce(lambda a,b:a+b)

    # make a new rdd
    # (methylationIndex,methylationType,posInRef,refName)
    # further processing
    methylationRddSecond = sc.parallelize(findMethylationFirst)

    # map:((methylationIndex,methylationType,posInRef,refName),1)
    # reduceByKey: ((methylationIndex,methylationType,posInRef,refName),number)
    disposeMethylationThird = methylationRddSecond\
        .map(lambda (a, b, c, d): ((a, b, c, d), 1))\
        .reduceByKey(lambda a, b: a + b)


    # count the position in every chromosome
    # map(refName, [posInRef])
    result_position = disposeMethylationThird\
        .map(lambda ((a, b, c, d),num): (d, [c]))\
        .reduceByKey(lambda a,b: a+b)\
        .map(lambda (a,b): (a, sorted(b)))\
        .collect()

    # count the methylation number in every chromosome
    # map((refName, posInRef), [(methylationType, number)])
    # map: return((refName, posInRef), [(the sequence is methylation_in_total, CHG(X),
    # CHH(H), CpG(Z), (CN or CHN)(U), TOTAL)])
    # map(refName, (posInRef, (total_methylation_num, X_num, H_num, Z_num, U_num, total_num)))
    result_arrangement = disposeMethylationThird\
        .map(lambda ((a, b, c, d), num): ((d, c), [(b, num)])) \
        .reduceByKey(lambda a, b: a + b) \
        .map(arrangementNumberPerPos)\
        .map(lambda ((a, b), c): (a, [(b, c)]))

    # map: sort the value by position in the chromosome
    # return (refName, list[(posInRef,(number))])
    result_number = result_arrangement\
        .reduceByKey(lambda a, b: a+b)\
        .map(lambda (a, b): (a, sorted(b, key=lambda c: c[0])))\
        .collect()


    # count the methylation PDF in every chromosome
    # map: return (chromosome,[(position,(total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf))])
    result_pdf = result_arrangement\
        .map(calculatePdf)\
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda (a, b): (a, sorted(b, key=lambda c: c[0])))\
        .collect()


    # Combine the records: result_position, result_number, result_pdf
    combinedRecords = combineThreeResults(result_position, result_number, result_pdf)

    # print combinedRecords
    # Save the records.
    saveRecords(outputFileAddress, None, combinedRecords)

    # build data structure for csv output format
    # result_arrangement: (refName, (posInRef, (total_methylation_num, X_num, H_num, Z_num, U_num, total_num)))
    # map: return (chromosome,[[position, totalNum, totalMeyNum, total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf]])
    # map: sort the list sequence according to the position
    # map: for each chromosome, save one csv file
    result_csv = result_arrangement\
        .map(calculatePdfCsv)\
        .reduceByKey(lambda a, b: a+b)\
        .map(lambda (a, b): (a, sorted(b, key=lambda c: c[0])))\
        .map(saveRecordsAsCsv)\
        .collect()


    print "Methylation analysing done sucscessfully!"



def calculatePdfCsv(record):
    # calculate the pdf of total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf respectively
    # return (chromosome, [[position, totalNum, totalMeyNum, total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf]])

    data = record[1]
    total_methylation_pdf = 0
    X_pdf = 0
    H_pdf = 0
    Z_pdf = 0
    U_pdf = 0
    tmp = 0

    total_num = data[0][1][5]
    total_methylation_num = data[0][1][0]

    if total_num != 0:

        tmp = float(data[0][1][0])/float(total_num)
        total_methylation_pdf = round(tmp, 2)
        tmp = float(data[0][1][1]) / float(total_num)
        X_pdf = round(tmp, 2)
        tmp = float(data[0][1][2]) / float(total_num)
        H_pdf = round(tmp, 2)
        tmp = float(data[0][1][3]) / float(total_num)
        Z_pdf = round(tmp, 2)
        tmp = float(data[0][1][4]) / float(total_num)
        U_pdf = round(tmp, 2)

    return ((record[0], [[record[1][0][0], total_num, total_methylation_num, total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf]]))


def calculatePdf(record):
    # calculate the pdf of total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf respectively
    data = record[1]
    total_methylation_pdf = 0
    X_pdf = 0
    H_pdf = 0
    Z_pdf = 0
    U_pdf = 0
    tmp = 0

    total_num = data[0][1][5]
    total_methylation_num = data[0][1][0]

    if total_num != 0:

        tmp = float(data[0][1][0])/float(total_num)
        total_methylation_pdf = round(tmp, 2)
        tmp = float(data[0][1][1]) / float(total_num)
        X_pdf = round(tmp, 2)
        tmp = float(data[0][1][2]) / float(total_num)
        H_pdf = round(tmp, 2)
        tmp = float(data[0][1][3]) / float(total_num)
        Z_pdf = round(tmp, 2)
        tmp = float(data[0][1][4]) / float(total_num)
        U_pdf = round(tmp, 2)

    return ((record[0],[(record[1][0][0],(total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf))]))


def arrangementNumberPerPos(record):
    # arrange the sequence of all kinds of results as follows: methylation_in_total, CHG(X),
    # CHH(H), CpG(Z), (CN or CHN)(U), TOTAL
    data = record[1]

    X_num=0
    x_num=0
    H_num=0
    h_num=0
    Z_num=0
    z_num=0
    U_num=0
    u_num=0
    total_num=0
    total_methylation_num=0

    i=0
    while i < len(data):
        if data[i][0] == 'X':
            X_num=data[i][1]
        elif data[i][0] == 'x':
            x_num=data[i][1]
        elif data[i][0] == 'H':
            H_num=data[i][1]
        elif data[i][0] == 'h':
            h_num=data[i][1]
        elif data[i][0] == 'Z':
            Z_num=data[i][1]
        elif data[i][0] == 'z':
            z_num=data[i][1]
        elif data[i][0] == 'U':
            U_num=data[i][1]
        elif data[i][0] == 'u':
            u_num=data[i][1]
        else:
            print "wrong!"
        i+=1

    total_num = X_num + x_num + H_num + h_num + Z_num + z_num + U_num + u_num
    total_methylation_num = X_num + H_num + Z_num + U_num

    return ((record[0],(total_methylation_num, X_num, H_num, Z_num, U_num, total_num)))


def filterRecordType(record):
    # The key is a tuple containing the name and type of the record.
    (name, dataType) = record[0]

    # Make sure the record and and type are correct.
    if dataType == "protobuf:lm.io.Methylation":
        return True
    print
    "there is a unknown datatype record: " + dataType
    return False


def checkStrandType(xm, xr, xg):
    # strand='+' if the reads is forward, else, strand='-'
    # index represents the type of four strands:OT, CTOT, OB, CTOB
    strand = ""
    index = 0

    # Determine strand and index
    if xm != None:
        # get rid of the front of xm
        xm_pure=xm[5:]
        # original top strand
        if xr == 'XR:Z:CT' and xg == 'XG:Z:CT':
            index = 0
            strand = '+'
        # complementary to original top strand
        elif xr == 'XR:Z:GA' and xg == 'XG:Z:CT':
            index = 1
            strand = '-'
        # complementary to original bottom strand
        elif xr == 'XR:Z:GA' and xg == 'XG:Z:GA':
            index = 2
            strand = '+'
        # original bottom strand
        elif xr == 'XR:Z:CT' and xg == 'XG:Z:GA':
            index = 3
            strand = '-'
        else:
            #sys.exit(1)
            print "wrong!"

        # reverse the methylation call if the read has been reverse-complemented for the output
        if strand == '-':
            xmToList = list(xm_pure)
            xmToList.reverse()
            xm_pure = "".join(xmToList)

    return (strand, index, xm_pure)


def findAllMethylationRelated(strand, index, newXm, cigar, pos, refName):
    # Parsing cigar string
    if cigar != None:
        # storing the length per operation
        length = re.split(r'[\D+]', cigar)
        # storing the operation
        operation = re.split(r'[\d+]', cigar)

        # remove the empty element
        while '' in length:
            length.remove('')
        while '' in operation:
            operation.remove('')

        # judge the cigar
        if len(length) != len(operation):
            print "CIGAR string contained a non-matching number of lengths and operations\n"
            #sys.exit(1)

        # build a extended cigar list
        i = 0
        cigarList = []
        while i < len(length):
            n = 0
            while n < int(length[i]):
                cigarList.append(operation[i])
                n += 1
            i += 1

    # Ajust the start position for all reads mapping to the reverse strand
    # Reverse strand's start position is located in 3'
    if strand == '-':
        # Reverse the cigarList for '-' reads
        cigarList.reverse()
        # linear match
        if re.match(r'^(\d+)M$', cigar) != None:
            cigarSplit = cigar.split("M")
            pos += int(cigarSplit[0]) - 1
        # InDel read
        else:
            count = 0
            for ops in cigarList:
                # Matching bases or deletions affect the genomic position of the 3' ends of reads, insertions don't
                if ops == 'M' or ops == 'D':
                    count += 1
            pos += count - 1

    # Find the specific posion
    methylationList = findPos(cigarList, pos, strand, newXm, refName)

    return methylationList


def searchXm(posInCigar, refPosOffset, xmPosOffset, xm, strand, pos):
    # search the XM and find the bases related to methylation according the position and information of cigar
    '''
    . for bases not involving cytosines
    X for methylated C in CHG context (was protected)
    x for not methylated C in CHG context (was converted)
    H for methylated C in CHH context (was protected)
    h for not methylated C in CHH context (was converted)
    Z for methylated C in CpG context (was protected)
    z for not methylated C in CpG context (was converted)
    u - C in Unknown context (CN or CHN) - unmethylated
    U - C in Unknown context (CN or CHN) - methylated
    '''

    methylationIndex = ''
    methylationType = ''
    posInRef = -1

    base = xm[posInCigar - xmPosOffset]
    # judge the methylation type and index
    if base == 'X':
        methylationIndex = '+'
        methylationType = 'X'
    elif base == 'x':
        methylationIndex = '-'
        methylationType = 'x'
    elif base == 'H':
        methylationIndex = '+'
        methylationType = 'H'
    elif base == 'h':
        methylationIndex = '-'
        methylationType = 'h'
    elif base == 'Z':
        methylationIndex = '+'
        methylationType = 'Z'
    elif base == 'z':
        methylationIndex = '-'
        methylationType = 'z'
    elif base == 'U':
        methylationIndex = '+'
        methylationType = 'U'
    elif base == 'u':
        methylationIndex = '-'
        methylationType = 'u'
    elif base == '.':
        return ('', '', -1)
    else:
        print
        "wrong xm input!\n"
        sys.exit(1)

    if strand == '+':
        # calculate the position in the reference
        posInRef = pos + posInCigar - refPosOffset

    elif strand == '-':
        # calculate the position in the reference
        posInRef = pos - posInCigar + refPosOffset

    return (methylationIndex, methylationType, posInRef)


def findPos(cigarList, pos, strand, newXm, refName):
    # methylated Cs (any context) will receive a forward (+) orientation
    # not methylated Cs (any context) will receive a reverse (-) orientation

    # traverse all the cigarList, determine the base's position according to the operation
    i = 0
    # refPosOffset is the offset when using cigar position(i) to locate the base in reference
    refPosOffset = 0
    # xmPosOffset is the offset when using cigar position(i) to locate the base in xm
    xmPosOffset = 0
    # a list contains all bases information related to methylation
    methylationList = []

    while i < len(cigarList):

        operation = cigarList[i]

        if operation == 'M':
            # nothing needs to be changed
            (methylationIndex, methylationType, posInRef) = searchXm(i, refPosOffset, xmPosOffset, newXm, strand, pos)
            # add this base to methylation list if the base is related with methylation
            if posInRef != -1:
                methylationList.append((methylationIndex, methylationType, posInRef, refName))
        elif operation == 'I':
            # insert means the bases in xm do not exist in the reference
            refPosOffset += 1
            (methylationIndex, methylationType, posInRef) = searchXm(i, refPosOffset, xmPosOffset, newXm, strand, pos)
            # add this base to methylation list if the base is related with methylation
            if posInRef != -1:
                methylationList.append((methylationIndex, methylationType, posInRef, refName))
        elif operation == 'D':
            # delete means the bases in reference do not exist in the xm
            xmPosOffset += 1
            refPosOffset -= 1
        else:
            print "wrong type of operation in cigar!\n"
            #sys.exit(1)

        i += 1

    return methylationList


def extractAlignmentInfo(record):
    # Extract the name, dataType, data from sfile
    (name, dataType) = record[0]
    data = record[1]

    # Deserialize the data.
    obj = Methylation_pb2.Methylation()
    obj.ParseFromString(str(data))

    refName = obj.rname.encode("utf-8")
    pos = obj.pos
    cigar = obj.cigar.encode("utf-8")
    xm = obj.xm.encode("utf-8")
    xr = obj.xr.encode("utf-8")
    xg = obj.xg.encode("utf-8")

    # Determine the reads derection and type of strands
    (strand, index, newXm) = checkStrandType(xm, xr, xg)

    # Find all the base related to methylation and calculate their position in relevant chromosome
    methylationList=findAllMethylationRelated(strand, index, newXm, cigar, pos, refName)

    return methylationList


def combineThreeResults(result_position, result_number, result_pdf):

    # making arrays to store all data
    # pos: 1 dimension
    # num: 3 dimensions (len,2,6) (posInRef, (total_methylation_num, X_num, H_num, Z_num, U_num, total_num))
    # pdf: 3 dimensions (len,2,5) (position, (total_methylation_pdf, X_pdf, H_pdf, Z_pdf, U_pdf))

    # combinedRecords: store all the dic format data (key is chromosome/(position)/(num)/(pdf))
    combinedRecords={}
    '''
    for chromosome in result_position:

        allPosPerChromosome = np.zeros((len(chromosome[1])))
        allPosPerChromosome[:] = chromosome[1]
        combinedRecords["/%s/Position" % (chromosome[0])] = allPosPerChromosome


    for chromosome in result_number:

        allNumPerChromosome = np.zeros((len(chromosome[1]), 2, 6))
        for i in range(0, len(chromosome[1])):
            allNumPerChromosome[:0:] = chromosome[1][i][0]
            allNumPerChromosome[:1:] = chromosome[1][i][1]
        combinedRecords["/%s/Numbers" % (chromosome[0])] = allNumPerChromosome


    for chromosome in result_pdf:

        allPDFPerChromosome = np.zeros((len(chromosome[1]), 2, 5))
        for i in range(0, len(chromosome[1])):
            allPDFPerChromosome[:0:] = chromosome[1][i][0]
            allPDFPerChromosome[:1:] = chromosome[1][i][1]
        combinedRecords["/%s/PDF" % (chromosome[0])] = allPDFPerChromosome
    '''

    for chromosome in result_position:

        allPosPerChromosome = np.zeros((len(chromosome[1])),dtype = int)
        allPosPerChromosome[:] = chromosome[1]
        combinedRecords["/%s/Position" % (chromosome[0])] = allPosPerChromosome


    for chromosome in result_number:

        allNumPerChromosome = np.zeros((len(chromosome[1]), 6),dtype = int)
        for i in range(0, len(chromosome[1])):
            allNumPerChromosome[i,:] = chromosome[1][i][1]
        combinedRecords["/%s/Numbers" % (chromosome[0])] = allNumPerChromosome


    for chromosome in result_pdf:

        allPDFPerChromosome = np.zeros((len(chromosome[1]), 5))
        for i in range(0, len(chromosome[1])):
            allPDFPerChromosome[i,:] = chromosome[1][i][1]
        combinedRecords["/%s/PDF" % (chromosome[0])] = allPDFPerChromosome

    return combinedRecords


def saveRecords(outputFilename, outputIndices, records):

    cellio.cellsave(outputFilename, records, indices=outputIndices, format="hdf5")
    print "Saved %d data sets to %s"%(len(records),outputFilename)


def saveRecordsAsCsv(record):
    # output results as csv format
    # column: position,total number related to methylation, total number of methylated bases,
    #         methylation PDF, X type PDF, H type PDF, Z type PDF, U type PDF
    global output_dir

    filename = record[0]
    result = record[1]
    columnName = ['position', 'totalNum', 'methyNum', 'methyPDF', 'X_PDF', 'H_PDF', 'Z_PDF', 'U_PDF' ]
    file = pd.DataFrame(columns = columnName, data = result)
    file.to_csv(output_dir.value+'/'+filename+'.csv')

    return 1


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Error: the number of arguments should be two, the outputAddress, the inputFile and the readsNum"
        quit()

    outputFileAddress = sys.argv[1]  # the address of the local output file to create
    inputFileName = sys.argv[2]  # the input SFile in HDFS
    #readsNum = sys.argv[3]  # the minimum number of reads in one position

    conf = SparkConf().setAppName("Methylation_Test")
    sc = SparkContext(conf=conf)
    main_spark(sc, outputFileAddress, inputFileName)

