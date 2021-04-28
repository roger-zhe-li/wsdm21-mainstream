# Leave No User Behind: Towards Improving the Utility of Recommender Systems for Non-mainstream Users

This is our implementation and experimental data for the paper:

Roger Zhe Li, Julián Urbano, Alan Hanjalic (2021). Leave No User Behind: Towards Improving the Utility of Recommender Systems for Non-mainstream Users. In Proceedings of WSDM '21, Virtual Event, Israel, March 8-12, 2021.

**Please cite our WSDM '21 paper if you use our code and data. Thanks!** 

Author: Roger Zhe Li (http://www.zhe-li.me)

## Environment Settings
We use PyTorch 1.6.0 as the main deep learning framework for implementation. The data analysis relies heavily on pyGAM.



## Example to run the code.
The instruction of commands has been clearly stated in the code (see the parse_args function). 

Run NAECF and DeepCoNN:

python3 test.py --dataset instant_video --coef_u 0.5  --coef_i 0.5  --seed 1992  --batch_size 256  --mode 4

Run MF:

python3 test_mf.py --dataset instant_video --seed 1992  --batch_size 256  --mode 5


### Dataset
We provide three processed datasets: Amazon Instant Video, Amazon Digital Music and BeerAdvocate. The BeerAdvocate dataset keeps users with at least 5 interactions, and sample 25% users in the original dataset.


### Cite

Please cite our WSDM'21 paper if you use the code.

```
@inproceedings{li2021leave,
  title={Leave No User Behind: Towards Improving the Utility of Recommender Systems for Non-mainstream Users},
  author={Li, Roger Zhe and Urbano, Juli{\'a}n and Hanjalic, Alan},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  pages={103--111},
  year={2021}
}

```


## License
* The paper is licensed under a [Creative Commons Attribution International 4.0 License](https://creativecommons.org/licenses/by/4.0/).
* Databases and their contents are distributed under the terms of the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
* Software is distributed under the terms of the [MIT License](https://opensource.org/licenses/MIT).



Last Update Date: January 21, 2021
