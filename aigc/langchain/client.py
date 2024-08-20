"""
Author: jhzhu
Date: 2024/8/17
Description: 
"""
import requests
import unittest


class ClientTest(unittest.TestCase):
    def test_post_method(self):
        response = requests.post(
            "http://localhost:8000/joke/invoke",
            json={'input': {'topic': 'cats'}}
        )
        print(response.json()['output']['content'])


