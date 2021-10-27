$(function () {
    $.get('/prediction_dynamic').done(function (result) {
        if (result['list'] != undefined) {
            real_list = result['real']
            data_list = result['list'];
            date_list = result['dates'];
            extent_date_list = result['extent dates'];
            console.log(extent_date_list.length);
            var myChart = echarts.init(document.getElementById('dynamic-figure'));
            var option = {
                title: {
                text: 'Passenger Flow Prediction',
                left: 'center'
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'cross',
                        label: {
                            backgroundColor: '#283b56'
                        }
                    }
                },
                legend: {
                    data:['Real', 'Prediction'],
                    left: 'left'
                },
                toolbox: {
                    show: true,
                    feature: {
                        dataView: {readOnly: false},
                        restore: {},
                        saveAsImage: {}
                    }
                },
                dataZoom: {
                    show: false,
                    start: 0,
                    end: 100
                },
                xAxis: [
                    {
                        type: 'category',
                        boundaryGap: true,
                        data: (function (){
                            res = date_list.slice(0, real_list.length - 20);
                            return res;
                        })()
                    },
                    {
                        type: 'category',
                        boundaryGap: true,
                        data: (function (){
                            res = extent_date_list.slice(0, real_list.length - 20);
                            return res;
                        })()
                    }
                ],
                yAxis: [
                    {
                        type: 'value',
                        name: 'Real',
                        top: '5%',
                        scale: true,
                        min: 0,
                        boundaryGap: [0.2, 0.2]
                    },
                    {
                        type: 'value',
                        name: 'Prediction',
                        top: '5%',
                        scale: true,
                        min: 0,
                        boundaryGap: [0.2, 0.2]
                    }
                ],
                series: [
                    {
                        name: 'Real',
                        type: 'bar',
                        xAxisIndex: 1,
                        yAxisIndex: 1,
                        data: (function (){
                            res = real_list.slice(0, real_list.length - 20)
                            return res;
                        })()
                    },
                    {
                        name: 'Prediction',
                        type: 'line',
                        data: (function (){
                            res = data_list.slice(0, real_list.length - 20);
                            return res;
                        })()
                    }
                ]
            }
        };

        appCount = real_list.length - 19;
        setInterval(function (){
            var data0 = option.series[0].data;
            var data1 = option.series[1].data;
            data0.shift();
            data0.push(real_list[appCount]);
            data1.shift();
            data1.push(data_list[appCount]);
            option.xAxis[0].data.shift();
            option.xAxis[0].data.push(date_list[appCount]);
            option.xAxis[1].data.shift();
            option.xAxis[1].data.push(extent_date_list[appCount]);
            appCount++;
            myChart.setOption(option);
        }, 500);
    });
});