# Runs the training of the various CVN networks
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk

import config
import data
import models
import utils

# main()
def main():
    print("\nCHIPS CVN - It's Magic\n")

    args = utils.parse_args()
    conf = config.process_config(args.config)

    # Create the train, val and test datasets...
    train_ds = data.DataHandler(args.train, conf)
    val_ds = data.DataHandler(args.train, conf)
    test_ds = data.DataHandler(args.train, conf)

    if conf.type == "pid":
        model = models.PIDModel(conf)
    elif conf.type == "ppe":
        model = models.PPEModel(conf)
    elif conf.type == "par":
        model = models.ParModel(conf)

    model.build()
    model.plot()
    model.summary()
    model.compile()

if __name__ == '__main__':
    main()
