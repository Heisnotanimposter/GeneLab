Columns in a VCF File:

	1.	#CHROM: The chromosome number where the variation is located.
	2.	POS: The position of the variation on the chromosome.
	3.	ID: An identifier for the variant, usually an rsID from dbSNP if known.
	4.	REF: The reference base(s) at the given position in the reference genome.
	5.	ALT: The alternate base(s) at this position (the variation).
	6.	QUAL: A quality score for the variation.
	7.	FILTER: Any filters that the variation fails (e.g., PASS if it passes all filters).
	8.	INFO: Additional information about the variation.
	9.	FORMAT: Format of the data in the subsequent individual genotype columns.
	10.	Sample Data: One or more columns, each representing the data for one individual. The content of these columns depends on the FORMAT column and might include genotype (GT), genotype quality (GQ), and depth of coverage (DP).
