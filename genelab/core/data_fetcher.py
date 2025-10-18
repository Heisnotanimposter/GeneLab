"""Genetic data fetcher for retrieving sequences from NCBI."""

import time
import logging
from typing import Optional
from Bio import Entrez, SeqIO

logger = logging.getLogger(__name__)


class GeneticDataFetcher:
    """Fetch genetic sequences from NCBI databases."""

    def __init__(self, email: str, api_key: Optional[str] = None, max_retries: int = 3, delay: float = 1.0):
        """
        Initialize the genetic data fetcher.

        Args:
            email: Email address for NCBI Entrez (required by NCBI)
            api_key: Optional NCBI API key for higher rate limits
            max_retries: Maximum number of retry attempts
            delay: Delay between requests in seconds
        """
        if not email:
            raise ValueError("Email is required for NCBI Entrez")

        self.email = email
        self.api_key = api_key
        self.max_retries = max_retries
        self.delay = delay

        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        logger.info(f"GeneticDataFetcher initialized with email: {email}")

    def fetch_sequence(self, accession: str, database: str = "nucleotide") -> Optional[str]:
        """
        Fetch a genetic sequence from NCBI.

        Args:
            accession: NCBI accession number
            database: Database to query (default: "nucleotide")

        Returns:
            Sequence string or None if fetch fails
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching sequence {accession} from {database} (attempt {attempt + 1})")
                handle = Entrez.efetch(db=database, id=accession, rettype="fasta", retmode="text")
                record = SeqIO.read(handle, "fasta")
                handle.close()

                sequence = str(record.seq)
                logger.info(f"Successfully fetched sequence {accession} (length: {len(sequence)})")
                return sequence

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay)
                else:
                    logger.error(f"Failed to fetch sequence {accession} after {self.max_retries} attempts")
                    return None

        return None

    def fetch_multiple_sequences(self, accessions: list[str], database: str = "nucleotide") -> dict[str, Optional[str]]:
        """
        Fetch multiple genetic sequences from NCBI.

        Args:
            accessions: List of NCBI accession numbers
            database: Database to query (default: "nucleotide")

        Returns:
            Dictionary mapping accession to sequence
        """
        results = {}
        for accession in accessions:
            sequence = self.fetch_sequence(accession, database)
            results[accession] = sequence
            time.sleep(self.delay)  # Be polite to NCBI servers

        return results

    def fetch_sequence_info(self, accession: str, database: str = "nucleotide") -> Optional[dict]:
        """
        Fetch metadata about a genetic sequence.

        Args:
            accession: NCBI accession number
            database: Database to query (default: "nucleotide")

        Returns:
            Dictionary with sequence metadata or None if fetch fails
        """
        try:
            logger.info(f"Fetching metadata for {accession}")
            handle = Entrez.esummary(db=database, id=accession, retmode="xml")
            record = Entrez.read(handle)
            handle.close()

            logger.info(f"Successfully fetched metadata for {accession}")
            return dict(record[0])

        except Exception as e:
            logger.error(f"Failed to fetch metadata for {accession}: {e}")
            return None
