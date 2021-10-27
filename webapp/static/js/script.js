$(function() {
    var myChart = echarts.init(document.getElementById('stop_visualize'));
    var myChartRoute = echarts.init(document.getElementById('route_visualize'));
    var myChartStopRoute = echarts.init(document.getElementById('stop_route_visualize'));
    var stopDataAxis = [];
    var stopData = [];
    var routeDataAxis = [];
    var routeData = [];
    var stopRouteDataAxis = [];
    var stopRouteData = [];
    var originalOption = {
        title: {
        text: 'Route Visualization'
        },
        tooltip: {},
        dataZoom: [
            {
                type: 'inside'
            }
        ],
        legend: {
            data:['Passenger Flow']
        },
        xAxis: {
            data: []
        },
        yAxis: {},
        series: [{
            name: 'Passenger Flow',
            type: 'bar',
            data: []
        }]
    };

    var option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        toolbox: {
            left: 'right',
            show: true,
            feature: {
                dataView: {readOnly: true},
                restore: {},
                saveAsImage: {}
            }
        },
        dataZoom: [
            {
                type: 'inside'
            }
        ],
        legend: {
            left: 'left',
            data:['Passenger Flow']
        },
        yAxis: {
            name: 'Passenger Number',
            type: 'value'
        },
        graphic: [
        {
            type: 'group',
            left: 'right',
            top: '5%',
            children: [
            {
                type: 'rect',
                z: 100,
                left: 'center',
                top: 'middle',
                shape: {
                    width: 290,
                    height: 100
                },
                style: {
                    fill: '#fff',
                    stroke: '#555',
                    lineWidth: 2,
                    shadowBlur: 8,
                    shadowOffsetX: 3,
                    shadowOffsetY: 3,
                    shadowColor: 'rgba(0,0,0,0.3)'
                }
            },
            {
                type: 'text'
                    }
                ]
            }
        ]
    };

    myChart.setOption(originalOption);
    myChartRoute.setOption(originalOption);
    myChartStopRoute.setOption(originalOption);

    myChart.setOption({
        title: {
        text: 'Stop Visualization'
        }
    });

    myChartRoute.setOption({
        title: {
        text: 'Route Visualization'
        }
    });

    myChartStopRoute.setOption({
        title: {
        text: 'Stop Route Visualization'
        }
    });

    $.get('/submit_item').done(function (result) {
        submit_item = result['submit item']
        $(".chart-frame").hide(); //Hide all content
        $(".tabs li").attr("id",""); //Reset id's
        if (submit_item == "None" || submit_item == "stop") {
            $(".tabs li:first").attr("id","current"); // Activate this
            $('#chart-stop').fadeIn(); // Show content for current tab
            myChart.showLoading();
            if (submit_item == "None") {
                $.get('/initialize').done(function (result) {
                if (result['list'] != undefined) {
                    myChart.hideLoading();
                    stopData = result['list'];
                    stopDataAxis = result['dates'];
                    myChart.setOption(option);
                    myChart.setOption({
                        title: {
                        left: 'center',
                        text: result['title'],
                        subtext: 'Time Interval: 1 Day'
                        },
                        xAxis: {
                            name: 'Date-time',
                            type: 'category',
                            data: result['dates']
                        },
                        graphic: [
                        {
                            type: 'group',
                            left: 'right',
                            top: '5%',
                            children: [
                            {
                                type: 'rect'
                            },
                            {
                                type: 'text',
                                z: 100,
                                left: 'center',
                                top: 'middle',
                                style: {
                                    fill: '#000',
                                    text: [
                                        'Max Value: '+ result['max'] + ' (' + result['max index'] + ')',
                                        'Min Value: '+ result['min'] + ' (' + result['min index'] + ')',
                                        'Mean Value: '+ result['mean'],
                                    ].join('\n\n'),
                                    font: '15px Microsoft YaHei'
                                        }
                                    }
                                ]
                            }
                        ],
                        series: [{
                            name: 'Passenger Flow',
                            type: 'bar',
                            data: result['list']
                        }]
                    })
                }
            });
            } else if (submit_item == "stop") {
                $.get('/stop_transfer').done(function (result) {
                if (result['list'] != undefined) {
                    myChart.hideLoading();
                    stopData = result['list'];
                    stopDataAxis = result['dates'];
                    myChart.setOption(option);
                    myChart.setOption({
                        title: {
                            left: 'center',
                            text: 'Bus Stop ID: ' + result['title'],
                            subtext: 'Time Interval: ' + result['time_interval']
                        },
                        xAxis: {
                            name: 'Date-time',
                            type: 'category',
                            data: result['dates']
                        },
                        graphic: [
                        {
                            type: 'group',
                            left: 'right',
                            top: '5%',
                            children: [
                            {
                                type: 'rect',
                            },
                            {
                                type: 'text',
                                z: 100,
                                left: 'center',
                                top: 'middle',
                                style: {
                                    fill: '#000',
                                    text: [
                                        'Max Value: '+ result['max'] + ' (' + result['max index'] + ')',
                                        'Min Value: '+ result['min'] + ' (' + result['min index'] + ')',
                                        'Mean Value: '+ result['mean'],
                                    ].join('\n\n'),
                                    font: '15px Microsoft YaHei'
                                        }
                                    }
                                ]
                            }
                        ],
                        series: [{
                            name: 'Passenger Flow',
                            type: 'bar',
                            data: result['list']
                        }]
                    })
                }
            })
            }
        };

        if (submit_item == "route") {
            $(".tabs li:nth-child(2)").attr("id","current"); // Activate this
            $('#chart-route').fadeIn(); // Show content for current tab
            myChartRoute.showLoading();
            $.get('/route_transfer').done(function (result) {
                myChartRoute.hideLoading();
                if (result['list'] != undefined) {
                    routeData = result['list'];
                    routeDataAxis = result['dates'];
                    myChartRoute.setOption(option);
                    myChartRoute.setOption({
                        color: ['#3398DB'],
                            title: {
                            text: 'Route : ' + result['title'],
                            left: 'center',
                            subtext: 'Time Interval: ' + result['time_interval']
                        },
                        xAxis: {
                            name: 'Date-time',
                            type: 'category',
                            data: result['dates']
                        },
                        graphic: [
                        {
                            type: 'group',
                            left: 'right',
                            top: '5%',
                            children: [
                            {
                                type: 'rect',
                            },
                            {
                                type: 'text',
                                z: 100,
                                left: 'center',
                                top: 'middle',
                                style: {
                                    fill: '#000',
                                    text: [
                                        'Max Value: '+ result['max'] + ' (' + result['max index'] + ')',
                                        'Min Value: '+ result['min'] + ' (' + result['min index'] + ')',
                                        'Mean Value: '+ result['mean'],
                                    ].join('\n\n'),
                                    font: '15px Microsoft YaHei'
                                        }
                                    }
                                ]
                            }
                        ],
                        series: [{
                            name: 'Passenger Flow',
                            type: 'bar',
                            data: result['list']
                        }]
                    })
                }
            })
        };

        if (submit_item == "stop_route") {
            $(".tabs li:last").attr("id","current");
            $('#chart-stop-route').fadeIn();
            myChartStopRoute.showLoading();
            $.get('/stop_route_transfer').done(function (result) {
                myChartStopRoute.hideLoading();
                if (result['list'] != undefined) {
                    stopRouteData = result['list'];
                    stopRouteDataAxis = result['dates'];
                    myChartStopRoute.setOption(option);
                    myChartStopRoute.setOption({
                        color: ['#D7DA8B'],
                            title: {
                            text: 'Route : ' + result['title_route'] + '   Stop ID: ' + result['title_stop'],
                            left: 'center',
                            subtext: 'Time Interval: ' + result['time_interval']
                        },
                        xAxis: {
                            name: 'Date-time',
                            type: 'category',
                            data: result['dates']
                        },
                        graphic: [
                        {
                            type: 'group',
                            left: 'right',
                            top: '5%',
                            children: [
                            {
                                type: 'rect',
                            },
                            {
                                type: 'text',
                                z: 100,
                                left: 'center',
                                top: 'middle',
                                style: {
                                    fill: '#000',
                                    text: [
                                        'Max Value: '+ result['max'] + ' (' + result['max index'] + ')',
                                        'Min Value: '+ result['min'] + ' (' + result['min index'] + ')',
                                        'Mean Value: '+ result['mean'],
                                    ].join('\n\n'),
                                    font: '15px Microsoft YaHei'
                                        }
                                    }
                                ]
                            }
                        ],
                        series: [{
                            name: 'Passenger Flow',
                            type: 'bar',
                            data: result['list']
                        }]
                    })
                }
            })
        }
    });

    var zoomSize = 6;

    myChart.on('click', function (params) {
        console.log(stopDataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)]);
        myChart.dispatchAction({
            type: 'dataZoom',
            startValue: stopDataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)],
            endValue: stopDataAxis[Math.min(params.dataIndex + zoomSize / 2, stopData.length - 1)]
        })
    });

    myChartRoute.on('click', function (params) {
        console.log(routeDataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)]);
        myChartRoute.dispatchAction({
            type: 'dataZoom',
            startValue: routeDataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)],
            endValue: routeDataAxis[Math.min(params.dataIndex + zoomSize / 2, routeData.length - 1)]
        })
    });

    myChartStopRoute.on('click', function (params) {
        console.log(stopRouteDataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)]);
        myChartStopRoute.dispatchAction({
            type: 'dataZoom',
            startValue: stopRouteDataAxis[Math.max(params.dataIndex - zoomSize / 2, 0)],
            endValue: stopRouteDataAxis[Math.min(params.dataIndex + zoomSize / 2, stopRouteData.length - 1)]
        })
    })
});

