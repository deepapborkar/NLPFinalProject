# Assignment 9 Submission

## Student Info
Deepa Borkar \
dborkar@uchicago.edu

## Instructions

The following are the endpoints of concern for this assignment: 
- Endpoint 1: http://dborkar-a9-web.ucmpcs.org:5000/annotate
- Endpoint 2: http://dborkar-a9-web.ucmpcs.org:5000/annotate/job

Please use Endpoint 1 in order to get to Endpoint 2. An error message will appear if Endpoint 2 is visited without the appropriate URL parameters. The annotator instance no longer has flask endpoints and is only used for processing messages from SQS.

## Description

The description is provided in detail in the A9 assignment instructions. I tried to handle any critical errors by adding error handling to the a9_web_server.py, a9_annotator.py, and a9_run.py files. 

## Resources

I put comments in the python files and html files to explain which references I relied on for help with the assignment. Please refer to the comments section for references.
