import re
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

class NERDatasetCreator:
    def __init__(
        self,
        text_column,
        entity_columns,
        tokenizer_name=None,
        max_length=512,
    ):
        """
        Initializes the NERDatasetCreator with the required text and entity columns.

        Args:
            text_column (str): The name of the column containing text data.
            entity_columns (dict): A dictionary where keys are column names in the DataFrame
                                   corresponding to entities, and values are their respective labels.
            tokenizer_name (str): The name of the BERT tokenizer to use.
        """
        self.text_column = text_column
        self.entity_columns = entity_columns
        if tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_length=max_length)
        else:
            self.tokenizer = None
        
        self.id_column = "savsnet_id"

    def create_ner_dataset(self, df):
        """
        Generate an NER dataset by extracting entities from each row of the DataFrame.
        Works with any tokenizer that can provide character-level offsets.

        Args:
            df (pd.DataFrame): DataFrame containing text and entity columns.

        Returns:
            pd.DataFrame: A DataFrame containing the id, sentence, and extracted entities.
        """
        ner_dataset = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            original_sentence = row[self.text_column]
            entities = []

            # Iterate over specified entity columns to extract entities
            for column, entity_label in self.entity_columns.items():
                sentence = original_sentence
                entity_values = str(row[column])

                # Skip if the entity column is empty
                if pd.isna(entity_values) or entity_values == "":
                    continue
                
                # sort entity_values by length to avoid overlapping entities
                entity_values = sorted(entity_values.split(","), key=len, reverse=True)

                # Remove duplicates 
                entity_values = list(dict.fromkeys(entity_values))
                
                # Process each entity in a comma-separated list
                for entity in entity_values:
                    entity = entity.strip()
                    
                    # Find all occurrences of the entity in the sentence
                    sentence, entity_positions = self._find_entity_positions(sentence, entity)
                    
                    for start_char, end_char in entity_positions:
                        entities.append({
                            "entity": entity,
                            "label": entity_label,
                            "start": start_char,
                            "end": end_char
                        })

            # Sort entities by start position
            entities = sorted(entities, key=lambda x: x["start"])

            # Append the row data to the NER dataset
            ner_dataset.append({
                "id": row.get(self.id_column, None),
                "sentence": original_sentence,
                "entities": entities
            })

        return pd.DataFrame(ner_dataset)

    def _find_entity_positions(self, text, entity):
        """
        Find all occurrences of an entity in the text and return their character positions.

        Args:
            text (str): The input text to search in.
            entity (str): The entity to search for.

        Returns:
            tuple: Updated text with entity masked and a list of (start, end) positions.
        """
        if not entity:  # Prevent infinite loop if entity is empty
            return text, []

        positions = []
        start = 0

        while True:
            start = text.find(entity, start)
            if start == -1:  # No more occurrences found
                break

            end = start + len(entity)

            # Optional: Ensure whole word match
            is_word_boundary = True
            if start > 0 and text[start - 1].isalnum():
                is_word_boundary = False
            if end < len(text) and text[end].isalnum():
                is_word_boundary = False

            if is_word_boundary:
                positions.append((start, end))

            start += len(entity)  # Ensure forward movement

        # Replace entity with a placeholder to avoid overlapping matches
        text = re.sub(re.escape(entity), "X" * len(entity), text)

        return text, positions