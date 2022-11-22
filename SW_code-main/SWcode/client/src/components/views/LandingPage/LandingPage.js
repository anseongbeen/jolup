
import { useSelector } from 'react-redux';
import React, { useEffect, useState } from 'react'
import { FaCode } from "react-icons/fa";
import axios from "axios";
import { Icon, Col, Card, Row, Carousel } from 'antd';
import Meta from 'antd/lib/card/Meta';
import ImageSlider from '../../utils/ImageSlider';
import Checkbox from './Sections/CheckBox';
import Radiobox from './Sections/RadioBox';
import SearchFeature from './Sections/SearchFeature';
import { Categories, spots } from './Sections/Datas';
import 고봉밥 from '../../../고봉밥.png';
import 로고봉밥 from '../../../로고봉밥.png';

function LandingPage() {

    const [Products, setProducts] = useState([])
    const [Skip, setSkip] = useState(0)
    const [Limit, setLimit] = useState(8)
    const [PostSize, setPostSize] = useState(0)
    const [Filters, setFilters] = useState({
        Categories: [],
        spots: []
    })
    const [SearchTerm, setSearchTerm] = useState("")


    let productNumber = {};
    let pId = {};
    let pTitle = {};
    useEffect(() => {

        let body = {
            skip: Skip,
            limit: Limit
        }

        getProducts(body)

    }, [])


    const userData = useSelector(state => state.user.userData)

        console.log(userData)


    axios.post('/api/users/number', userData)
        .then(response => {
            if(response.data.success){
                productNumber = response.data.pNumber;
                pId = response.data.product_id;
                pTitle = response.data.product_title;
                const mukza = window.confirm('오늘의 추천 가게 : ' + pTitle + '\n추천 가게로 이동하시겠습니까?')
                if(mukza){
                    window.location.replace('http://localhost:3000/product/' + pId);
                }
            }
        })

    const getProducts = (body) => {
        axios.post('/api/product/products', body)
            .then(response => {
                if (response.data.success) {
                    if (body.loadMore) {
                        setProducts([...Products, ...response.data.productInfo])
                    } else {
                        setProducts(response.data.productInfo)
                    }
                    setPostSize(response.data.postSize)
                } else {
                    alert(" 상품들을 가져오는데 실패 했습니다.")
                }
            })
    }


    



    const loadMoreHanlder = () => {

        let skip = Skip + Limit
        let body = {
            skip: skip,
            limit: Limit,
            loadMore: true,
            filters: Filters
        }

        getProducts(body)
        setSkip(skip)
    }


    const renderCards = Products.map((product, index) => {
        if(product.price !== 0){
        return <Col lg={6} md={8} xs={24} key={index}>
            <Card
                cover={<a href={`/product/${product._id}`} ><ImageSlider images={product.images} /></a>}
            >
                <Meta
                    title={<a href={`/product/${product._id}`} style={{ color: 'rgb(50,50,50)' }} >{product.title}</a>}
                />
            </Card>
        </Col>
        }
    })

    const showFilteredResults = (filters) => {

        let body = {
            skip: 0,
            limit: Limit,
            filters: filters
        }

        getProducts(body)
        setSkip(0)

    }


    const handleSpots = (value) => {
        const data = spots;
        let array = [];

        for (let key in data) {
            if (data[key]._id === parseInt(value, 10)) {
                array = data[key].array;
            }
        }
        return array;
    }

    const handleFilters = (filters, category) => {

        const newFilters = { ...Filters }

        newFilters[category] = filters

        console.log('filters', filters)

        if (category === "spots") {
            let spotsValues = handleSpots(filters)
            newFilters[category] = spotsValues
        }
        showFilteredResults(newFilters)
        setFilters(newFilters)
    }

    const updateSearchTerm = (newSearchTerm) => {

        let body = {
            skip: 0,
            limit: Limit,
            filters: Filters,
            searchTerm: newSearchTerm
        }

        setSkip(0)
        setSearchTerm(newSearchTerm)
        getProducts(body)

    }



    return (
        <div style={{ width: '75%', margin: '3rem auto' }}>

            <div style={{ textAlign: 'center' }}>
                <h2><a><img src={고봉밥} alt="고봉밥" /></a> </h2>
            </div>

            {/* Filter */}

            <Row gutter={[16, 16]}>
                <Col lg={12} xs={24}>
                    {/* CheckBox */}
                    <Checkbox list={Categories} handleFilters={filters => handleFilters(filters, "Categories")} />
                </Col>
                <Col lg={12} xs={24}>
                    {/* RadioBox */}
                    <Radiobox list={spots} handleFilters={filters => handleFilters(filters, "spots")} />
                </Col>
            </Row>





            {/* Search */}

            <div style={{ display: 'flex', justifyContent: 'flex-end', margin: '1rem auto' }}>
                <SearchFeature
                    refreshFunction={updateSearchTerm}
                />
            </div>

            {/* Cards */}


            <Row gutter={[16, 16]} >
                {renderCards}
            </Row>

            <br />

            {PostSize >= Limit &&
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <button onClick={loadMoreHanlder}>더보기</button>
                </div>
            }

        </div>
    )
}

export default React.memo(LandingPage);