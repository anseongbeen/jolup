const express = require('express');
const router = express.Router();
const { User } = require("../models/User");
const { Product } = require("../models/Product");
const { Payment } = require("../models/Payment");

const { auth } = require("../middleware/auth");
const async = require('async');

//=================================
//             User
//=================================

router.get("/auth", auth, (req, res) => {
    res.status(200).json({
        _id: req.user._id,
        isAdmin: req.user.role === 0 ? false : true,
        isAuth: true,
        email: req.user.email,
        name: req.user.name,
        lastname: req.user.lastname,
        role: req.user.role,
        uCounter: req.user.uCounter,
        image: req.user.image,
        cart: req.user.cart,
        recommend: req.user.recommend,
        history: req.user.history
    });
});

router.post("/register", (req, res) => {

    const user = new User(req.body);

    user.save((err, doc) => {
        if (err) return res.json({ success: false, err });
        return res.status(200).json({
            success: true
        });
    });
});

router.post("/login", (req, res) => {
    User.findOne({ email: req.body.email }, (err, user) => {
        if (!user)
            return res.json({
                loginSuccess: false,
                message: "Auth failed, email not found"
            });

        user.comparePassword(req.body.password, (err, isMatch) => {
            if (!isMatch)
                return res.json({ loginSuccess: false, message: "Wrong password" });

            user.generateToken((err, user) => {
                if (err) return res.status(400).send(err);
                res.cookie("w_authExp", user.tokenExp);
                res
                    .cookie("w_auth", user.token)
                    .status(200)
                    .json({
                        loginSuccess: true, userId: user._id
                    });
            });
        });
    });
});

router.get("/logout", auth, (req, res) => {
    User.findOneAndUpdate({ _id: req.user._id }, { token: "", tokenExp: "" }, (err, doc) => {
        if (err) return res.json({ success: false, err });
        return res.status(200).send({
            success: true
        });
    });
});


router.post("/addToCart", auth, (req, res) => {

    //??????  User Collection??? ?????? ????????? ????????? ???????????? 
    User.findOne({ _id: req.user._id },
        (err, userInfo) => {

            // ????????? ???????????? ???????????? ????????? ?????? ????????? ?????? ?????? ????????? ?????? 

            let duplicate = false;
            userInfo.cart.forEach((item) => {
                if (item.id === req.body.productId) {
                    duplicate = true;
                }
            })

            //????????? ?????? ?????????
            if (duplicate) {
                User.findOneAndUpdate(
                    { _id: req.user._id, "cart.id": req.body.productId },
                    { $inc: { "cart.$.quantity": 1 } },
                    { new: true },
                    (err, userInfo) => {
                        if (err) return res.status(200).json({ success: false, err })
                        res.status(200).send(userInfo.cart)
                    }
                )
            }
            //????????? ?????? ?????? ????????? 
            else {
                User.findOneAndUpdate(
                    { _id: req.user._id },
                    {
                        $push: {
                            cart: {
                                id: req.body.productId,
                                quantity: 1,
                                date: Date.now()
                            }
                        }
                    },
                    { new: true },
                    (err, userInfo) => {
                        if (err) return res.status(400).json({ success: false, err })
                        res.status(200).send(userInfo.cart)
                    }
                )
            }
        })
});


router.get('/removeFromCart', auth, (req, res) => {

    //?????? cart?????? ?????? ???????????? ??? ????????? ???????????? 
    User.findOneAndUpdate(
        { _id: req.user._id },
        {
            "$pull":
                { "cart": { "id": req.query.id } }
        },
        { new: true },
        (err, userInfo) => {
            let cart = userInfo.cart;
            let array = cart.map(item => {
                return item.id
            })

            //product collection??????  ?????? ???????????? ???????????? ????????? ???????????? 

            //productIds = ['5e8961794be6d81ce2b94752', '5e8960d721e2ca1cb3e30de4'] ??????????????? ????????????
            Product.find({ _id: { $in: array } })
                .populate('writer')
                .exec((err, productInfo) => {
                    return res.status(200).json({
                        productInfo,
                        cart
                    })
                })
        }
    )
})






const func = (uCounter) => {
    return new Promise((resolve, reject) => {
        
        const spawn = require('child_process').spawn; 

        const result_01 = spawn('python', ['test.py', uCounter])
        result_01.stdout.on('data', (result)=>{
            return resolve(result);
        });
    })
}

router.post('/number', async (req, res) => {
    console.log(req.body)
    console.log("?????????~~~~~~~~~~~~~~")
    let funcResult = await func(req.body.uCounter);
    console.log("funcResult: " + funcResult);

    const num1 = funcResult;
    let test = ''
    //console.log("1 : " + test)
    Product.findOne({views: num1})
            .exec((err, products) => {
                if (err) return res.status(400).send(err);
                test = products._id
                //console.log("2 : "+test)
                res.status(200).json({ 
                    success: true, 
                    pNumber: products.views,
                    product_id: products._id,
                    product_title: products.title
                })
                
            })
            User.findOneAndUpdate(
                {_id: req.body._id},
                //?????? ???????????? - test
            )
    //console.log("3 : " + test)

    // User.findOneAndUpdate(
    //     { _id: req.body._id },
    //     {
    //         $set: {
    //             recommend: {
    //                 id: 
    //             }
    //         }
    //     },
    //     { new: true },
    //     (err) => {
    //         if (err) return res.status(400).json({ success: false, err })
    //         res.status(200).json({ success: true })
    //     }
    // )
});


module.exports = router;
