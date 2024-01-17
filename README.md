# Summarisation test-bench

Here you  can find a summarisation test-bench that specifically focuses on summarisations of at most 50 words. 
The metrics used are Rouge-L and BLEU. This test-bench was used for a paper titled 'Summariser text-generation models in a conversational context'.



## Usage:

### Installation:

1. Clone the repository using ``git clone [soon]``
2. If you want to use custom data, edit the ``json_file_path`` in ``utils.py`` or modify the ``samples/samples/data_prefixes.json``.

### Using the program 

You can run the program using ``python .\main.py [arguments]``. Valid arguments are:
- --test_bleu (will generate BLEU test scores)
- --test_rouge (will generate Rouge-L test scores)
- --output_to_file (will output the results to a .csv file - SOON)
- --simple (Just run the already set-up models with both BLEU and Rouge-L scores)
- --models <model> <model> <model> (Specifies one or more models)

Examples:

``python .\main.py --simple``

``python .\main.py --test_bleu --models "sshleifer/distilbart-cnn-12-6"``

``python .\main.py --test_rouge --models "sshleifer/distilbart-cnn-12-6" "google/pegasus-xsum"``
